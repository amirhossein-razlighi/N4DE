from render import Renderer
import numpy as np
import torch
from models_animated import SDFModule
from utils import *


if __name__ == "__main__":
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
        f="data/sphere.pt",
    ).cuda()

    renderer = Renderer(num_views=10, res=512, fname=None)

    for frame_num in range(3):
        with torch.no_grad():
            vertices_np, faces_np = model.get_zero_points(t=frame_num)
            v = vertices_np.shape[0]
            f = faces_np.shape[0]
            vertices.data[:v] = torch.from_numpy(vertices_np)
            faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            face_normals = compute_face_normals(vertices[:v], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v], faces[:f], face_normals
            )
        est_imgs = renderer.render(vertices[:v], faces[:f], vertex_normals)

        save_image(
            f"data/test_sphere_init/{frame_num:06d}_est.png",
            est_imgs[0, ..., :3].detach().cpu().numpy(),
        )
