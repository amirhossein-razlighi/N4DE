import torch
from models_animated import EncoderPlusSDF, RenderHead
import numpy as np
from rendering_gaussian import render, Camera, render_with_seperate_args
from render import Renderer
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main():
    sdf_head = EncoderPlusSDF().cuda()
    sdf_head.load_state_dict(torch.load("checkpoints/model.pth"))
    sdf_head.eval()

    render_head = RenderHead(3).cuda()
    optimizer = torch.optim.Adam(render_head.parameters(), lr=1e-3)

    EPOCHS = 200
    NUM_SAMPLES = int(200 * 200 * 200)
    BATCH_SIZE = 2_000_000

    t = 0
    vertices_np, faces_np = sdf_head.get_zero_points(
        mesh_res=200,
        t=t,
        device="cuda",
        batch_size=20_000,
    )
    v = vertices_np.shape[0]
    concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
    vertices = torch.from_numpy(concated).float().cuda()
    faces = torch.from_numpy(np.copy(faces_np)).int().contiguous().cuda()

    face_normals = compute_face_normals(vertices[:, :3], faces)
    vertex_normals = compute_vertex_normals(vertices[:, :3], faces, face_normals)

    with torch.no_grad():
        R = Renderer(50, 256, dr.RasterizeGLContext(), "data/Bunny/1.obj")
        targets = R.render(vertices[..., :3], faces[..., :3], vertex_normals)

    cameras = []
    for i in range(50):
        camera = Camera(
            R.fov_x,
            R.fov_x,
            256,
            256,
            R.view_mats[i],
            R.mvps[i],
            R.camera_positions[i],
        )
        cameras.append(camera)

    best_loss = torch.inf
    pbar = tqdm(range(EPOCHS * 5))

    for epoch in pbar:
        vertices[..., 3:] = torch.rand((vertices.shape[0], 1)).cuda()
        optimizer.zero_grad()
        sh_coeffs = []
        opacity = []
        scaling = []
        rotation = []

        idx = 0
        while idx < vertices.shape[0]:
            end_idx = min(idx + BATCH_SIZE, vertices.shape[0])
            sh, op, sc, rot = render_head(vertices[idx:end_idx].detach())
            sh_coeffs.append(sh)
            opacity.append(op)
            scaling.append(sc)
            rotation.append(rot)
            idx += BATCH_SIZE

        sh_coeffs = torch.cat(sh_coeffs, dim=0).cuda()
        opac = torch.cat(opacity, dim=0).cuda()
        scaling = torch.cat(scaling, dim=0).cuda()
        rotation = torch.cat(rotation, dim=0).cuda()

        rand_cam = np.random.randint(0, 50)
        camera = cameras[rand_cam]
        est = render_with_seperate_args(
            camera,
            vertices[..., :3],
            3,
            opac,
            scaling,
            rotation,
            sh_coeffs,
            torch.tensor([0.0, 0.0, 0.0]).cuda(),
        )["render"].permute(1, 2, 0)

        loss = torch.nn.functional.l1_loss(est, targets[rand_cam][..., :3])
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            pbar.set_description(f"Epoch :{epoch} | Loss: {loss.item()}")
            torch.save(render_head.state_dict(), "checkpoints/render_init.pth")
            plt.imsave("test.png", est.cpu().detach().numpy().clip(0, 1))


if __name__ == "__main__":
    main()
