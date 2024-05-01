import numpy as np
import torch
import os
import sys
import trimesh
import torch
import torch.nn as nn

import nvdiffrast.torch as dr

from pytorch_lightning.utilities.seed import seed_everything
from PIL import Image
import utils

seed_everything(42)


class Renderer:
    def __init__(self, num_views, res, fname=None, scale=1.75, device="cuda:0"):
        self.num_views = num_views
        self.num_views = num_views
        self.res = res
        self.r_mvp = []
        self.r_campos = []
        self.mvs = []
        self.lightdir = []

        self.glctx = dr.RasterizeGLContext()
        self.zero_tensor = torch.as_tensor(0.0, dtype=torch.float32, device=device)
        proj = utils.projection(x=0.5, n=1.5, f=100.0)
        self.fov_x = np.rad2deg(2 * np.arctan(0.5))

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
        self.rots = []

        for i in range(num_views):
            r_rot = utils.random_rotation()
            r_mv = np.matmul(utils.translate(0, 0, -4.0), r_rot)
            self.mvs.append(r_mv)
            r_mvp = np.matmul(proj, r_mv).astype(np.float32)
            self.r_mvp.append(r_mvp)
            r_campos = torch.as_tensor(
                np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device=device
            )
            lightdir = -r_campos / torch.norm(r_campos)
            self.lightdir.append(lightdir)

        proj = torch.as_tensor(proj, dtype=torch.float32, device=device)
        self.view_mats = torch.as_tensor(
            np.array(self.mvs), dtype=torch.float32, device=device
        )
        self.lightdir = torch.stack(self.lightdir)
        self.mvps = proj @ self.view_mats
        self.render_target(fname, device)

    def render_target(self, fname, device="cuda:0"):
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
            normals, dtype=torch.float32, device=device
        ).contiguous()

        v = torch.as_tensor(
            mesh.vertices, dtype=torch.float32, device=device
        ).contiguous()
        f = torch.as_tensor(mesh.faces, dtype=torch.int32, device=device).contiguous()

        self.target_imgs = self.render(v, f, normals, device)

    def render_pointlight(self, pos, pos_idx, normals, device="cuda:0"):
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))
        depth_ = v_ndc[..., 2]
        depth = depth_.unsqueeze(-1).contiguous()

        rast, _ = dr.rasterize(self.glctx, v_ndc, pos_idx, [self.res, self.res])
        v_cols = torch.zeros_like(pos).to(device)

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
        return torch.nan_to_num(result)

    def render(self, pos, pos_idx, normals, device="cuda:0"):
        return self.render_pointlight(pos, pos_idx, normals, device)


if __name__ == "__main__":
    R = Renderer(5, 512, fname="data/bunny.obj")
    print(f"Shape is {R.target_imgs.shape}")
    for i in range(R.target_imgs.shape[0]):
        utils.save_image(
            f"data/test_render/{i:06d}.png",
            R.target_imgs[i, :, :, :3].detach().cpu().numpy(),
        )
        utils.save_image(
            f"data/test_render/{i:06d}_depth.png",
            R.target_imgs[i, :, :, -1].detach().cpu().numpy(),
        )
