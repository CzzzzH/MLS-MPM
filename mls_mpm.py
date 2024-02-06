import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

dim = 3
neighbour = (3, ) * dim
bound = 3

n_grid = 128
dx = 1. / n_grid
inv_dx = float(n_grid)
dt = 1e-4
substeps = int(1 / 120 // dt)

p_rho = 1
p_vol = (dx * 0.5) ** dim
p_mass = p_vol * p_rho
g = 9.8

E = 5e3
nu = 0.2
mu_0 = E / (2. * (1. + nu))
lam_0 = E * nu / ((1. + nu) * (1. - 2. * nu))
Ef = 400

bunny_vertices = []
bunny_scale = 1.5
bunny_pos = np.array([0.5, 0.2, 0.5])

spot_vertices = []
spot_scale = 0.1
spot_pos = np.array([0.3, 0.5, 0.5])

armadillo_vertices = []
armadillo_scale = 0.2
armadillo_pos = np.array([0.7, 0.5, 0.5])

with open("bunny.obj", "r") as f:
    for line in f:
        if line[0] != "v":
            break
        tokens = line[:-1].split(" ")
        bunny_vertices.append(np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])]))

with open("spot.obj", "r") as f:
    for line in f:
        if line[0] != "v":
            continue
        tokens = line[:-1].split(" ")
        spot_vertices.append(np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])]))

with open("armadillo.obj", "r") as f:
    for line in f:
        if line[0] != "v":
            continue
        tokens = line[:-1].split(" ")
        armadillo_vertices.append(np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])]))

n_bunny = len(bunny_vertices)
n_spot = len(spot_vertices)
n_armadillo = len(armadillo_vertices)
n_particles = n_bunny + n_spot + n_armadillo

print(f"Bunny Particles Num: {n_bunny}")
print(f"Spot Particles Num: {n_spot}")
print(f"Armadillo Particles Num: {n_armadillo}")
print(f"Total Particles Num: {n_particles}")

x = ti.Vector.field(dim, dtype=float, shape=(n_particles,))
v = ti.Vector.field(dim, dtype=float, shape=(n_particles,))
C = ti.Matrix.field(dim, dim, dtype=float, shape=(n_particles,))
F = ti.Matrix.field(dim, dim, dtype=float, shape=(n_particles,))
mat_idx = ti.field(dtype=int, shape=(n_particles,))
mat_color = ti.Vector.field(3, dtype=float, shape=(n_particles,))

Jp = ti.field(dtype=float, shape=(n_particles,))
Jf = ti.field(dtype=float, shape=(n_particles,))
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid,) * dim)
grid_m = ti.field(dtype=float, shape=(n_grid, ) * dim)

if dim == 3:
    for i in range(n_particles):
        
        if i < n_bunny:
            x[i] = bunny_vertices[i] * bunny_scale + bunny_pos
            mat_idx[i] = 0
            mat_color[i] = [0.0, 0.6, 1.0]
        elif i < n_bunny + n_spot:
            x[i] = spot_vertices[i - n_bunny] * spot_scale + spot_pos
            mat_idx[i] = 1
            mat_color[i] = [1.0, 0.6, 0.0]
        else:
            x[i] = armadillo_vertices[i - n_bunny - n_spot] * armadillo_scale + armadillo_pos
            mat_idx[i] = 2
            mat_color[i] = [0.0, 1.0, 0.0]
        
        F[i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        C[i] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        v[i] = [0, 0, 0]
        Jp[i] = 1
        Jf[i] = 1
else:
    for i in range(n_particles):
        
        if i < n_bunny:
            x[i] = bunny_vertices[i] * bunny_scale + bunny_pos
            mat_idx[i] = 0
            mat_color[i] = [0.0, 0.6, 1.0]
        elif i < n_bunny + n_spot:
            x[i] = spot_vertices[i - n_bunny] * spot_scale + spot_pos
            mat_idx[i] = 1
            mat_color[i] = [1.0, 0.6, 0.0]
        else:
            x[i] = armadillo_vertices[i - n_bunny - n_spot] * armadillo_scale + armadillo_pos
            mat_idx[i] = 2
            mat_color[i] = [0.0, 1.0, 0.0]

        F[i] = [[1, 0], [0, 1]]
        C[i] = [[0, 0], [0, 0]]
        v[i] = [0, 0]
        Jp[i] = 1
        Jf[i] = 1

@ti.kernel
def substep():

    for I in ti.grouped(grid_m):
        grid_v[I] = grid_v[I] * 0
        grid_m[I] = 0
    
    # P2G
    for p in x:
        
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        
        h = ti.exp(10 * (1.0 - Jp[p]))
        if mat_idx[p] == 1: # Elastic
            h = 1
        mu, lam = mu_0 * h, lam_0 * h
            
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            new_sig = sig[d, d]
            if mat_idx[p] == 2:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        
        affine = ti.Matrix.identity(float, dim)
        if mat_idx[p] == 0: # weakly compressible fluid
            stress = -dt * 4 * Ef * p_vol * (Jf[p] - 1) * inv_dx * inv_dx
            affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
        else:
            if mat_idx[p] == 2:
                F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * lam * J * (J - 1)
            stress = (-dt * 4  * p_vol * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
        
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
                 
    # Boundary   
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
            grid_v[I][1] -= dt * g
            for i in ti.static(range(dim)):
                if I[i] < bound and grid_v[I][i] < 0:
                    grid_v[I][i] = 0
                if I[i] > n_grid - bound and grid_v[I][i] > 0:
                    grid_v[I][i] = 0
    
    # G2P
    for p in x:
        
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = offset.cast(float) - fx
            g_v = grid_v[base + offset]
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p], C[p] = new_v, new_C
        Jf[p] *= 1 + dt * new_C.trace()
        x[p] += dt * v[p]

if __name__ == '__main__':
    
    animate_name = "mls_mpm"
    window = ti.ui.Window(animate_name, (1024, 1024), vsync=True)
    
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    output_flag = True
    frame = 0
    
    while window.running:
        
        current_t = 0.0
        for i in range(substeps):
            substep()
            current_t += dt
            
        camera.position(0.5, 0.5, -0.8)
        camera.lookat(0.5, 0.5, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 10, 0), color=(1, 1, 1))
        scene.ambient_light([0, 0, 0])

        scene.particles(centers=x, radius=0.003, per_vertex_color=mat_color)
        canvas.scene(scene)
        
        # if output_flag == True:
        #     window.save_image(f'output_{animate_name}/{animate_name}_{frame}.png')
        #     frame += 1
        
        window.show()