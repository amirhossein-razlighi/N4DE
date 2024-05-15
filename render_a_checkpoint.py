from render import Renderer
import numpy as np
import torch
from models_animated import SDFModule
from utils import *
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--mesh_dir", type=str, default=None)
    args = parser.parse_args()

    vertices = torch.zeros((3000000, 3)).cuda()
    normals = torch.zeros((3000000, 3)).cuda()
    faces = torch.empty((3000000, 3), dtype=torch.int32).cuda()

    model = SDFModule(
        cfg=Namespace(
            dim=4,
            out_dim=1,
            hidden_size=512,
            n_blocks=4,
            z_dim=1,
            const=60.0,
        ),
        f=args.checkpoint,
    ).cuda()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    render_targets = args.mesh_dir is not None
    if render_targets:
        renderers = []
        mesh_files = sorted(
            [
                os.path.join(args.mesh_dir, f)
                for f in os.listdir(args.mesh_dir)
                if f.endswith(".obj") or f.endswith(".ply")
            ]
        )
        for mesh_file in mesh_files:
            renderers.append(Renderer(num_views=10, res=512, fname=mesh_file))
    else:
        renderer = Renderer(num_views=10, res=512, fname=None)

    for frame_num in range(args.num_frames):
        with torch.no_grad():
            vertices_np, faces_np = model.get_zero_points(t=frame_num / 10, mesh_res=200)
            v = vertices_np.shape[0]
            f = faces_np.shape[0]
            vertices.data[:v] = torch.from_numpy(vertices_np)
            faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            face_normals = compute_face_normals(vertices[:v], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v], faces[:f], face_normals
            )
        if not render_targets:
            est_imgs = renderer.render(vertices[:v], faces[:f], vertex_normals)
        else:
            save_image(
                f"{args.output_dir}/{frame_num:06d}_gt.png",
                renderers[frame_num].target_imgs[0, ..., :3].detach().cpu().numpy(),
            )
            est_imgs = renderers[frame_num].render(
                vertices[:v], faces[:f], vertex_normals
            )
        
        img_to_save = est_imgs[0, ..., :3]
        save_image(
            f"{args.output_dir}/{frame_num:06d}_est.png",
            img_to_save.detach().cpu().numpy(),
        )
