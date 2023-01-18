import warp as wp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpm.mpm_model import MPMModelBuilder, mpm_collide, Mesh, DenseVolume
from mpm.mpm_integrator import (
    compute_grid_bound,
    set_grid_bound,
    g2p,
    p2g,
    grid_op,
    grid_op_with_contact,
    create_soft_contacts,
    eval_soft_contacts,
    MPMModel,
    MPMState,
    zero_everything,
    set_optim_variables,
    extract_grad_tensor,
)
import os


class MPMSimulator(torch.nn.Module):
    def __init__(self):
        super(MPMSimulator, self).__init__()

        wp.init()

        self.device = "cuda"
        self.sim_dt = 5e-4

        builder = MPMModelBuilder()
        builder.set_mpm_domain([1.0, 1.0, 1.0], 0.005)
        E = 1e5
        nu = 0.3
        ys = 10000.0
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        cohesion = 0.1
        friction_angle = np.pi / 3

        count = builder.add_mpm_grid(
            pos=(0.0, 0.2, 0.0),
            vel=(0.0, 0.0, 0.0),
            dim_x=30,
            dim_y=15,
            dim_z=30,
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            density=3e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=0,
            jitter=True,
            placement_x="center",
            placement_y="corner",
            placement_z="center",
            color=(125 / 255, 87 / 255, 0),
        )

        # This is for creating the initial shape for the particles.
        # Replace this with 0.5 initilization.
        import trimesh
        def trimesh2sdf(meshes, margin, dx, bbox=None):
            if meshes is None:
                return None
            mesh = trimesh.util.concatenate(meshes)

            if bbox is None:
                bbox = mesh.bounds.copy()

            sdfs = []
            normals = []
            for mesh in meshes:
                center = (bbox[0] + bbox[1]) / 2
                res = np.ceil((bbox[1] - bbox[0] + margin * 2) / dx).astype(int)
                lower = center - res * dx / 2.0

                points = np.zeros((res[0], res[1], res[2], 3))
                x = np.arange(0.5, res[0]) * dx + lower[0]
                y = np.arange(0.5, res[1]) * dx + lower[1]
                z = np.arange(0.5, res[2]) * dx + lower[2]

                points[..., 0] += x[:, None, None]
                points[..., 1] += y[None, :, None]
                points[..., 2] += z[None, None, :]

                points = points.reshape((-1, 3))

                query = trimesh.proximity.ProximityQuery(mesh)
                sdf = query.signed_distance(points) * -1.0

                surface_points, _, tri_id = query.on_surface(points)
                face_normal = mesh.face_normals[tri_id]
                normal = (points - surface_points) * np.sign(sdf)[..., None]
                length = np.linalg.norm(normal, axis=-1)
                mask = length < 1e6
                normal[mask] = face_normal[mask]
                normal = normal / (
                    np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
                )
                sdf = sdf.reshape(res)
                normal = normal.reshape((res[0], res[1], res[2], 3))

                sdfs.append(sdf)
                normals.append(normal)

            if len(sdfs) == 1:
                sdf = sdfs[0]
                normal = normals[0]
            else:
                sdfs = np.stack(sdfs)
                normals = np.stack(normals)
                index = np.expand_dims(sdfs.argmin(0), 0)
                sdf = np.take_along_axis(sdfs, index, 0)[0]
                normal = np.take_along_axis(normals, np.expand_dims(index, -1), 0)[0]

            return {
                "sdf": sdf,
                "normal": normal,
                "position": lower,
                "scale": np.ones(3) * dx,
                "dim": res,
            }

        box = trimesh.creation.box((0.05, 0.1, 0.05))
        sdf = trimesh2sdf([box], 0.02, 0.01)

        volume = DenseVolume(
            np.concatenate([sdf["normal"], sdf["sdf"][..., None]], -1),
            position=sdf["position"],
            scale=sdf["scale"],
            mesh=Mesh(box.vertices, box.faces.flatten()),
        )

        body = builder.add_body(wp.transform((0.0, 0.5, 0.0)))
        builder.add_shape_dense_volume(body, volume=volume, density=2e3)

        self.model = builder.finalize(self.device)

        self.model.struct.particle_radius = 0.005
        self.model.mpm_contact_distance = 0.005
        self.model.mpm_contact_margin = 0.01

        self.state_0 = self.model.state(requires_grad=True)
        self.state_1 = self.model.state(requires_grad=True)

        builder.clear_particles()
        builder.add_mpm_grid(
            pos=(0.0, 0.2, 0.0),
            vel=(0.0, 0.0, 0.0),
            dim_x=15,
            dim_y=15,
            dim_z=15,
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            density=3e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=0,
            jitter=True,
            placement_x="center",
            placement_y="corner",
            placement_z="center",
            color=(125 / 255, 87 / 255, 0),
        )
        builder.init_model_state(self.model, self.state_0)

    def forward(self, v):
        grid_m_ini = torch.ones([self.model.struct.grid_dim_x, 
                            self.model.struct.grid_dim_y, 
                            self.model.struct.grid_dim_z], 
                            dtype=torch.float32, 
                            device=self.device, 
                            requires_grad=True) * v
        
        final_grid_m = MPMSimulatorFunc.apply(self.model, self.state_0, self.state_1, self.sim_dt, self.device, grid_m_ini)
        return final_grid_m
        
class MPMSimulatorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, state_in, state_out, dt, device, grid_m_ini):

        n_steps = 20
        n_substeps = 30

        ctx.n_steps = n_steps
        ctx.n_substeps = n_substeps
        ctx.model = model
        ctx.device = device

        grid_m_seq = torch.zeros([n_steps, model.struct.grid_dim_x, model.struct.grid_dim_y, model.struct.grid_dim_z], dtype=torch.float32, device=device, requires_grad=True)

        for i_frame in range(0, n_steps):
            for i_step in range(0, n_substeps):

                state_in, state_out = state_out, state_in

                wp.launch(
                    zero_everything,
                    dim=int(
                        max(
                            model.body_count,
                            model.struct.n_particles,
                            model.struct.grid_dim_x
                            * model.struct.grid_dim_y
                            * model.struct.grid_dim_z,
                        )
                    ),
                    inputs=[
                        state_in.struct,
                        state_in.ext_body_f,
                        state_in.int_body_f,
                        state_in.mpm_contact_count,
                        model.struct.grid_dim_x,
                        model.struct.grid_dim_y,
                        model.struct.grid_dim_z,
                        model.struct.n_particles,
                        model.body_count,
                    ],
                    device=device,
                )

                wp.launch(
                    set_optim_variables,
                    dim=int(
                        model.struct.grid_dim_x
                        * model.struct.grid_dim_y
                        * model.struct.grid_dim_z
                    ),
                    inputs=[
                        state_in.struct,
                        wp.torch.from_torch(grid_m_ini),
                        model.struct.grid_dim_x,
                        model.struct.grid_dim_y,
                        model.struct.grid_dim_z,
                    ],
                    device=device,
                )

                wp.launch(
                    set_grid_bound,
                    dim=1,
                    inputs=[model.struct, state_in.struct],
                    device=device,
                )

                wp.launch(
                    create_soft_contacts,
                    dim=int(model.struct.n_particles * model.shape_count),
                    inputs=[
                        model.struct.n_particles,
                        state_in.struct.particle_q,
                        state_in.body_q,
                        model.shape_transform,
                        model.shape_body,
                        model.shape_geo_type,
                        model.shape_geo_id,
                        model.shape_geo_scale,
                        model.mpm_contact_margin,
                        state_in.mpm_contact_count,
                        state_in.mpm_contact_particle,
                        state_in.mpm_contact_body,
                        state_in.mpm_contact_body_pos,
                        state_in.mpm_contact_body_vel,
                        state_in.mpm_contact_normal,
                        model.mpm_contact_max,
                    ],
                    device=device,
                )

                wp.launch(
                    eval_soft_contacts,
                    dim=int(model.mpm_contact_max),
                    inputs=[
                        model.struct,
                        state_in.struct,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        state_in.mpm_contact_count,
                        state_in.mpm_contact_particle,
                        state_in.mpm_contact_body,
                        state_in.mpm_contact_body_pos,
                        state_in.mpm_contact_body_vel,
                        state_in.mpm_contact_normal,
                        model.struct.particle_radius,
                        state_in.ext_body_f,
                    ],
                    device=device,
                )

                wp.launch(
                    p2g,
                    dim=int(model.struct.n_particles),
                    inputs=[
                        model.struct,
                        state_in.struct,
                        state_out.struct,
                        model.gravity,
                        dt,
                    ],
                    device=device,
                )

                wp.launch(
                    grid_op_with_contact,
                    dim=int(
                        model.struct.grid_dim_x
                        * model.struct.grid_dim_y
                        * model.struct.grid_dim_z
                    ),
                    inputs=[
                        model.struct,
                        state_in.struct,
                        dt,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        model.shape_transform,
                        model.shape_body,
                        model.shape_geo_type,
                        model.shape_geo_id,
                        model.shape_geo_scale,
                        model.shape_count,
                        state_in.ext_body_f,
                    ],
                    device=device,
                )

                wp.launch(
                    g2p,
                    dim=int(model.struct.n_particles),
                    inputs=[model.struct, state_in.struct, state_out.struct, dt],
                    device=device,
                )

                final_grid_m = torch.zeros([model.struct.grid_dim_x, model.struct.grid_dim_y, model.struct.grid_dim_z], dtype=torch.float32, device=device, requires_grad=True)
                wp.launch(
                    extract_grad_tensor,
                    dim=int(
                        model.struct.grid_dim_x
                        * model.struct.grid_dim_y
                        * model.struct.grid_dim_z
                    ),
                    inputs=[
                        state_in.struct,
                        model.struct.grid_dim_x,
                        model.struct.grid_dim_y,
                        model.struct.grid_dim_z,
                    ],
                    outputs=[
                        wp.torch.from_torch(final_grid_m),
                    ],
                    device=device,
                )
            
            grid_m_seq[i_frame] = final_grid_m
            
            print(torch.argmax(torch.sum(torch.sum(final_grid_m, dim=0), dim=1)))
        
        ctx.grid_m_seq = grid_m_seq
        ctx.state_in = state_in

        return grid_m_seq

    @staticmethod
    def backward(ctx, grad_output):

        # n_steps = ctx.n_steps
        # n_substeps = ctx.n_substeps
        # model = ctx.model
        # device = ctx.device
        # grid_m_seq = ctx.grid_m_seq

        # state_in = ctx.state_in

        # for i_frame in reversed(range(0, n_steps)):
        #     for i_step in reversed(range(0, n_substeps)):

        #         adj_x = torch.zeros_like(ctx.x).contiguous()

        #         wp.launch(
        #             extract_grad_tensor,
        #             dim=int(
        #                 model.struct.grid_dim_x
        #                 * model.struct.grid_dim_y
        #                 * model.struct.grid_dim_z
        #             ),
        #             inputs=[
        #                 state_in.struct,
        #                 model.struct.grid_dim_x,
        #                 model.struct.grid_dim_y,
        #                 model.struct.grid_dim_z,
        #             ],
        #             outputs=[
        #                 None,
        #             ],
        #             adj_inputs=[wp.torch.from_torch(adj_x)],
        #             adj_outputs=[wp.torch.from_torch(grad_output)],
        #             device=device,
        #             adjoint=True
        #         )
                

        return grad_output

if __name__ == "__main__":

    v = torch.tensor(0.5).to("cuda").requires_grad_(True)

    simulator = MPMSimulator()
    grid_m_seq = simulator(v)
    # 
    print(grid_m_seq)
