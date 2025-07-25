import numpy as np
import torch
import trimesh
import torch

import nvdiffrast.torch as dr

from pytorch_lightning import seed_everything
import utils

seed_everything(42)


class Renderer:
    def __init__(
        self,
        num_views,
        res,
        glctx: dr.RasterizeGLContext,
        fname=None,
        scale=1.75,
        device="cuda:0",
        with_texture=False,
        random_cameras=False,
    ):
        self.num_views = num_views
        self.res = res
        self.device = device
        self.r_campos = []
        self.mvs = []
        self.lightdir = []

        self.glctx = glctx
        self.zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device=device)
        proj = utils.projection(x=0.5, n=1.5, f=100.0)
        self.fov_x = self.fov_y = np.rad2deg(2 * np.arctan(0.5))

        t = utils.translate(0, 0, 4.0)
        e = 1.5 / 0.5
        focal_length = (res / 2) / (1 / e)

        self.intrinsics = np.array(
            [
                [focal_length, 0.0, res / 2],
                [0.0, focal_length, res / 2],
                [0.0, 0.0, 1.0],
            ]
        )

        self.albedo = 0.55
        self.scale = scale

        for i in range(num_views):
            r_rot = utils.random_rotation()
            r_mv = np.matmul(utils.translate(0, 0, -4.0), r_rot)
            self.mvs.append(r_mv)
            r_campos = torch.as_tensor(
                np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device=device
            )
            self.r_campos.append(r_campos)
            lightdir = -r_campos / torch.norm(r_campos)
            self.lightdir.append(lightdir)

        if random_cameras:
            # set another set of random cameras for target views which are not necessarily the same as source views
            self.r_campos_target = []
            self.mvs_target = []
            self.lightdir_target = []

            for i in range(num_views):
                r_rot = utils.random_rotation()
                r_mv = np.matmul(utils.translate(0, 0, -4.0), r_rot)
                self.mvs_target.append(r_mv)
                r_campos = torch.as_tensor(
                    np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device=device
                )
                self.r_campos_target.append(r_campos)
                lightdir = -r_campos / torch.norm(r_campos)
                self.lightdir_target.append(lightdir)
        else:
            self.r_campos_target = self.r_campos
            self.mvs_target = self.mvs
            self.lightdir_target = self.lightdir

        proj = torch.as_tensor(proj, dtype=torch.float32, device=device)
        self.view_mats = torch.as_tensor(
            np.array(self.mvs), dtype=torch.float32, device=device
        )
        self.lightdir = torch.stack(self.lightdir)
        self.mvps = proj @ self.view_mats

        if random_cameras:
            self.view_mats_target = torch.as_tensor(
                np.array(self.mvs_target), dtype=torch.float32, device=device
            )
            self.lightdir_target = torch.stack(self.lightdir_target)
            self.mvps_target = proj @ self.view_mats_target
        else:
            self.view_mats_target = self.view_mats
            self.mvps_target = self.mvps
            self.lightdir_target = self.lightdir

        self.render_target(fname, device, with_texture)

    def load_texture(self, mesh, device):
        texture = np.array(mesh.visual.material.image.convert("RGB"))
        texture = torch.as_tensor(texture, dtype=torch.float32, device=device) / 255.0
        return texture

    def render_target(self, fname, device="cuda:0", with_texture=False):
        # Load Mesh
        if fname is not None:
            mesh = trimesh.load_mesh(fname)
        else:
            mesh = trimesh.load_mesh("data/bunny.obj")

        mean = np.mean(mesh.vertices, axis=0, keepdims=True)
        mesh.vertices -= mean
        scale = self.scale / (np.max(mesh.vertices) - np.min(mesh.vertices))
        mesh.vertices *= scale
        self.mesh = mesh

        normals = mesh.vertex_normals
        normals = torch.as_tensor(
            np.copy(normals), dtype=torch.float32, device=device
        ).contiguous()

        v = torch.as_tensor(
            np.copy(mesh.vertices), dtype=torch.float32, device=device
        ).contiguous()
        f = torch.as_tensor(
            np.copy(mesh.faces), dtype=torch.int32, device=device
        ).contiguous()

        if with_texture:
            try:
                uvs = mesh.visual.uv.copy()
                uvs = torch.as_tensor(
                    uvs, dtype=torch.float32, device=device
                ).contiguous()
                self.texture = self.load_texture(mesh, device)
                self.target_imgs = self.render(
                    v, f, normals, device, uvs=uvs, is_target=True
                )
            except:
                print("No texture found. Rendering without texture.")
                self.target_imgs = self.render(
                    v, f, normals, device=device, is_target=True
                )
        else:
            self.target_imgs = self.render(v, f, normals, device=device, is_target=True)

    def render_pointlight(
        self, pos, pos_idx, normals, device="cuda:0", is_target=False
    ):
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        if not is_target:
            v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        else:
            v_ndc = torch.matmul(v_hom, self.mvps_target.transpose(1, 2))
        depth_ = v_ndc[..., 2]
        depth = depth_.unsqueeze(-1).contiguous()

        rast, _ = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res, self.res])

        pixel_normals = dr.interpolate(normals[None, ...], rast, pos_idx)[0]
        if not is_target:
            diffuse = self.albedo * torch.sum(
                -self.lightdir.view(-1, 1, 1, 3) * pixel_normals, -1, keepdim=True
            )
        else:
            diffuse = self.albedo * torch.sum(
                -self.lightdir_target.view(-1, 1, 1, 3) * pixel_normals,
                -1,
                keepdim=True,
            )
        result = dr.antialias(
            torch.where(rast[..., -1:] != 0, diffuse, self.zero_tensor),
            rast,
            v_ndc,
            pos_idx,
        )
        depth_map, _ = dr.interpolate(depth, rast, pos_idx)
        result = result.repeat(1, 1, 1, 3)  # make it RGB
        result = torch.cat([result, depth_map], -1)  # add depth channel
        return torch.nan_to_num(result)

    def render_coloured(
        self,
        pos,
        pos_idx,
        normals,
        view_idx=None,
        vertex_colors=None,
        uvs=None,
        device="cuda:0",
        albedo=None,
    ):
        if albedo is None:
            albedo = self.albedo

        if vertex_colors is None and uvs is None:
            raise ValueError("Either vertex_colors or uvs must be provided")

        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        if view_idx is not None:
            v_ndc = v_hom @ self.mvps[view_idx].unsqueeze(0).transpose(1, 2)
        depth_ = v_ndc[..., 2]
        depth = depth_.unsqueeze(-1).contiguous()

        rast, _ = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res, self.res])

        pixel_normals = dr.interpolate(normals[None, ...], rast, pos_idx)[0]

        if uvs is not None:
            tex = self.texture[None, ...]
            texc, _ = dr.interpolate(uvs[None, ...], rast, pos_idx)
            pixel_colors = dr.texture(tex.contiguous(), texc)
        else:
            pixel_colors = dr.interpolate(vertex_colors[None, ...], rast, pos_idx)[0]

        if view_idx is not None:
            diffuse = albedo * torch.sum(
                -self.lightdir[view_idx].view(-1, 1, 3) * pixel_normals,
                -1,
                keepdim=True,
            )
        else:
            diffuse = albedo * torch.sum(
                -self.lightdir.view(-1, 1, 1, 3) * pixel_normals, -1, keepdim=True
            )
        diffuse = diffuse * pixel_colors
        result = dr.antialias(
            torch.where(rast[..., -1:] != 0, diffuse, self.zero_tensor),
            rast,
            v_ndc,
            pos_idx,
        )
        depth_map, _ = dr.interpolate(depth, rast, pos_idx)
        result = torch.cat([result, depth_map], -1)  # add depth channel
        return torch.nan_to_num(result)

    def render(
        self,
        pos,
        pos_idx,
        normals,
        vertex_colors=None,
        uvs=None,
        device="cuda:0",
        is_target=False,
    ):
        return (
            self.render_pointlight(pos, pos_idx, normals, device, is_target=is_target)
            if uvs is None and vertex_colors is None
            else (
                self.render_coloured(
                    pos,
                    pos_idx,
                    normals,
                    vertex_colors=vertex_colors,
                    uvs=uvs,
                    device=device,
                )
                if uvs is not None
                else self.render_coloured(
                    pos, pos_idx, normals, vertex_colors, uvs=None, device=device
                )
            )
        )

    @property
    def camera_positions(self):
        if not isinstance(self.r_campos, torch.Tensor):
            self.r_campos = torch.stack(self.r_campos).to(self.device)
        return self.r_campos

    @property
    def viewing_directions(self):
        viewing_directions = []
        for mv in self.mvs:
            # Extract the rotation part of the model-view matrix
            rotation_matrix = mv[:3, :3]
            # Viewing direction is the negative Z-axis transformed by the rotation matrix
            viewing_direction = np.dot(rotation_matrix, np.array([0, 0, -1]))
            viewing_directions.append(viewing_direction)
        return torch.as_tensor(
            viewing_directions, dtype=torch.float32, device=self.device
        )


class DatasetRenderer:
    def __init__(
        self,
        imgs,
        c2w,
        intrinsic,
        scale,
        offset,
        from_na: bool,
        device="cuda:0",
        config=None,
    ):
        self.device = device
        self.glctx = dr.RasterizeGLContext()
        self.zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device=device)
        self.target_imgs = imgs.to(device)
        self.res = imgs.shape[1]
        self.c2w = c2w.to(device)
        self.num_views = len(imgs)
        self.r_campos = []
        self.mvs = []
        self.lightdir = []
        self.intrinsics = intrinsic.to(device)
        self.albedo = 0.55
        self.scale = scale
        self.offset = offset

        config.num_views = self.num_views
        config.res = self.res

        ###### For reading from matrices in dataset #####

        # self.w2c = []
        # for i in range(self.num_views):
        #     w2c = c2w[i].inverse()
        #     w2c[:3, :3] = self.scale * w2c[:3, :3]
        #     w2c[:3, 3] = self.scale * w2c[:3, 3] + self.offset
        #     self.w2c.append(w2c.to(device))

        # proj = self.intrinsics

        # # Apply rotation if from_na is True
        # if from_na:
        #     rotation_matrix = torch.tensor(
        #         [
        #             [1, 0, 0, 0],
        #             [0, -1, 0, 0],
        #             [0, 0, -1, 0],
        #             [0, 0, 0, 1],
        #         ],  # rotate 180 degrees around x-axis
        #         dtype=torch.float32,
        #         device=device,
        #     ).to(device)
        #     self.w2c = [rotation_matrix @ w for w in self.w2c]

        # for i in range(self.num_views):
        #     r_mv = self.w2c[i]
        #     self.mvs.append(r_mv)
        #     r_campos = (self.w2c[i].inverse())[:3, 3]
        #     self.r_campos.append(r_campos)
        #     lightdir = -r_campos / torch.norm(r_campos)
        #     self.lightdir.append(lightdir)

        # self.view_mats = torch.stack(self.mvs)
        # self.lightdir = torch.stack(self.lightdir)
        # self.mvps = proj @ self.view_mats

        # self.fov_x = [
        #     np.rad2deg(2 * np.arctan(self.res / (2 * intrinsic_[0, 0].item())))
        #     for intrinsic_ in self.intrinsics
        # ]

        # self.fov_y = [
        #     np.rad2deg(2 * np.arctan(self.res / (2 * intrinsic_[1, 1].item())))
        #     for intrinsic_ in self.intrinsics
        # ]

        ##### Random cameras #####
        proj = utils.projection(x=0.5, n=1.5, f=100.0)
        self.fov_x = self.fov_y = np.rad2deg(2 * np.arctan(0.5))

        t = utils.translate(0, 0, 4.0)
        e = 1.5 / 0.5
        focal_length = (self.res / 2) / (1 / e)

        self.intrinsics = np.array(
            [
                [focal_length, 0.0, self.res / 2],
                [0.0, focal_length, self.res / 2],
                [0.0, 0.0, 1.0],
            ]
        )

        self.albedo = 0.55
        self.scale = scale
        self.rots = []

        for i in range(self.num_views):
            r_rot = utils.random_rotation()
            r_mv = np.matmul(utils.translate(0, 0, -4.0), r_rot)
            self.mvs.append(r_mv)
            r_campos = torch.as_tensor(
                np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device=device
            )
            self.r_campos.append(r_campos)
            lightdir = -r_campos / torch.norm(r_campos)
            self.lightdir.append(lightdir)

        proj = torch.as_tensor(proj, dtype=torch.float32, device=device)
        self.view_mats = torch.as_tensor(
            np.array(self.mvs), dtype=torch.float32, device=device
        )
        self.lightdir = torch.stack(self.lightdir)
        self.mvps = proj @ self.view_mats

    def render(
        self, pos, pos_idx, normals, vertex_colors=None, uvs=None, device="cuda:0"
    ):
        return (
            self.render_pointlight(pos, pos_idx, normals, device)
            if uvs is None and vertex_colors is None
            else (
                self.render_coloured(
                    pos,
                    pos_idx,
                    normals,
                    vertex_colors=vertex_colors,
                    uvs=uvs,
                    device=device,
                )
                if uvs is not None
                else self.render_coloured(
                    pos, pos_idx, normals, vertex_colors, uvs=None, device=device
                )
            )
        )

    def render_pointlight(self, pos, pos_idx, normals, device="cuda:0"):
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        depth_ = v_ndc[..., 2]
        depth = depth_.unsqueeze(-1).contiguous()

        rast, _ = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res, self.res])

        pixel_normals = dr.interpolate(normals[None, ...], rast, pos_idx)[0]
        diffuse = self.albedo * torch.sum(
            -self.lightdir.view(-1, 1, 1, 3) * pixel_normals, -1, keepdim=True
        )
        result = dr.antialias(
            torch.where(rast[..., -1:] != 0, diffuse, self.zero_tensor),
            rast,
            v_ndc,
            pos_idx,
        )
        depth_map, _ = dr.interpolate(depth, rast, pos_idx)
        result = result.repeat(1, 1, 1, 3)  # make it RGB
        result = torch.cat([result, depth_map], -1)  # add depth channel
        return torch.nan_to_num(result).float()

    def render_coloured(
        self,
        pos,
        pos_idx,
        normals,
        view_idx=None,
        vertex_colors=None,
        uvs=None,
        device="cuda:0",
        albedo=None,
    ):
        if albedo is None:
            albedo = self.albedo

        if vertex_colors is None and uvs is None:
            raise ValueError("Either vertex_colors or uvs must be provided")

        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        if view_idx is not None:
            v_ndc = v_hom @ self.mvps[view_idx].unsqueeze(0).transpose(1, 2)
        depth_ = v_ndc[..., 2]
        depth = depth_.unsqueeze(-1).contiguous()

        rast, _ = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res, self.res])

        pixel_normals = dr.interpolate(normals[None, ...], rast, pos_idx)[0]

        if uvs is not None:
            tex = self.texture[None, ...]
            texc, _ = dr.interpolate(uvs[None, ...], rast, pos_idx)
            pixel_colors = dr.texture(tex.contiguous(), texc)
        else:
            pixel_colors = dr.interpolate(vertex_colors[None, ...], rast, pos_idx)[0]

        if view_idx is not None:
            diffuse = albedo * torch.sum(
                -self.lightdir[view_idx].view(-1, 1, 3) * pixel_normals,
                -1,
                keepdim=True,
            )
        else:
            diffuse = albedo * torch.sum(
                -self.lightdir.view(-1, 1, 1, 3) * pixel_normals, -1, keepdim=True
            )
        diffuse = diffuse * pixel_colors
        result = dr.antialias(
            torch.where(rast[..., -1:] != 0, diffuse, self.zero_tensor),
            rast,
            v_ndc,
            pos_idx,
        )
        depth_map, _ = dr.interpolate(depth, rast, pos_idx)
        result = torch.cat([result, depth_map], -1)  # add depth channel
        return torch.nan_to_num(result)

    def visualize(self, vertices):
        import plotly.graph_objects as go

        # get one random camera
        cam_idx = np.random.randint(0, self.num_views)
        mvp = self.mvps[cam_idx]

        # get the vertices in homogeneous coordinates
        vertices_hom = torch.nn.functional.pad(vertices, (0, 1), "constant", 1.0)

        # transform the vertices to clip space
        vertices_clip = torch.matmul(vertices_hom, mvp.transpose(0, 1))

        # plot the original vertices and also the transformed vertices and camera position
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=vertices[:, 0].cpu().numpy(),
                y=vertices[:, 1].cpu().numpy(),
                z=vertices[:, 2].cpu().numpy(),
                mode="markers",
                marker=dict(size=2),
                name="Original Vertices",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=vertices_clip[:, 0].cpu().numpy(),
                y=vertices_clip[:, 1].cpu().numpy(),
                z=vertices_clip[:, 2].cpu().numpy(),
                mode="markers",
                marker=dict(size=2),
                name="Transformed Vertices",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=self.r_campos[cam_idx][0].cpu().numpy(),
                y=self.r_campos[cam_idx][1].cpu().numpy(),
                z=self.r_campos[cam_idx][2].cpu().numpy(),
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Camera Position",
            )
        )
        fig.write_html("temp.html")

    @property
    def camera_positions(self):
        if not isinstance(self.r_campos, torch.Tensor):
            self.r_campos = torch.stack(self.r_campos).to(self.device)
        return self.r_campos


if __name__ == "__main__":
    glctx = dr.RasterizeGLContext()
    R = Renderer(5, 512, fname="data/Static_Screaming_Face/1.obj", glctx=glctx)
    print(f"Shape is {R.target_imgs.shape}")
    for i in range(R.target_imgs.shape[0]):
        utils.save_image(
            f"Test/test_render/{i:06d}.png",
            R.target_imgs[i, :, :, :3].detach().cpu().numpy(),
        )
        utils.save_image(
            f"Test/test_render/{i:06d}_depth.png",
            R.target_imgs[i, :, :, -1].detach().cpu().numpy(),
        )
