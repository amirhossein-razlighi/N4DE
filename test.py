from models_animated import EncoderPlusSDF, RenderHead
from rendering_gaussian import render_with_seperate_args, Camera
import torch
import numpy as np
from utils import *
from render import Renderer
import trimesh
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from tqdm import tqdm

# model = EncoderPlusSDF().cuda()
# model.load_state_dict(torch.load("logs/5_DEFORMABLE_BREAKING_SPHERE/iter_0000000/model.pth"))
# model.eval()

# res = 200
# bound = -1.0

# xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
# grid = np.concatenate(
#     [ys[..., np.newaxis], xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1
# ).astype(np.float)
# grid = (grid / float(res - 1) - 0.5) * 2 * bound
# grid = grid.reshape(-1, 3)

# xyz = torch.tensor(grid).float().cuda()
# xyz = (xyz + 1) / 2
# xyz_posenc = positional_encoding(xyz, 64)

# with torch.no_grad():
#     offset = model.beta_predictor(xyz_posenc)
#     print(offset.abs().min())
#     print(offset.abs().max())
#     xyz_ = inverse_sigmoid(xyz) + offset
#     xyz_ = torch.sigmoid(xyz_)
#     print((xyz_ - xyz).abs().min())
#     print((xyz_ - xyz).abs().max())
#     print((xyz_ - xyz).mean())

# glctx = dr.RasterizeCudaContext()
# renderer = Renderer(1, 256, glctx, "data/Colored_Bunny/1.obj", with_texture=True)
# plt.imsave("bunny.png", renderer.target_imgs[-1].detach().cpu().numpy().clip(0, 1))


def sample_from_faces(vertices, faces, face_normals, num_samples, normal_offset=0.0):
    """
    Sample points from mesh faces with optional displacement along the face normals.

    Args:
        vertices (torch.Tensor): Vertex positions with shape (V, 3).
        faces (torch.Tensor): Face indices with shape (F, 3).
        face_normals (torch.Tensor): Normals of each face, shape (F, 3).
        num_samples (int): Number of points to sample.
        normal_offset (float): Distance to offset samples along the normal direction.

    Returns:
        sampled_points (torch.Tensor): Sampled points of shape (num_samples, 3).
    """
    # Step 1: Get vertices of each face
    face_vertices = vertices[faces]  # Shape: (F, 3, 3)

    # Step 2: Calculate areas of faces for sampling
    edge1 = face_vertices[:, 1] - face_vertices[:, 0]
    edge2 = face_vertices[:, 2] - face_vertices[:, 0]
    face_areas = 0.5 * torch.norm(
        torch.cross(edge1, edge2, dim=1), dim=1
    )  # Shape: (F,)

    # Step 3: Probabilistically select faces based on area
    face_probs = face_areas / face_areas.sum()
    selected_faces = torch.multinomial(face_probs, num_samples, replacement=True)

    # Step 4: Sample points within each selected face using barycentric coordinates
    u = torch.sqrt(torch.rand(num_samples, 1, device=vertices.device))
    v = torch.rand(num_samples, 1, device=vertices.device)
    w = 1 - u
    u, v = u * w, v * w  # Adjusted barycentric coordinates

    face_verts_selected = face_vertices[selected_faces]
    sampled_points = (
        u * face_verts_selected[:, 0]
        + v * face_verts_selected[:, 1]
        + (1 - u - v) * face_verts_selected[:, 2]
    )

    # Step 5: Offset sampled points along face normals if specified
    if normal_offset != 0.0:
        sampled_normals = face_normals[selected_faces]
        sampled_points += normal_offset * sampled_normals

    return sampled_points


# Preload SDF and Render Heads
sdf_head = EncoderPlusSDF().cuda()
sdf_head.load_state_dict(torch.load("logs/6_STATIC_COLORED_BUNNY_WITH_GS/model.pth"))
sdf_head.eval()

render_head = RenderHead(3).cuda()
render_head.load_state_dict(torch.load("checkpoints/render_init.pth"))
optimizer = torch.optim.Adam(render_head.parameters(), lr=1e-3)

EPOCHS = 200
NUM_SAMPLES = int(200 * 200 * 200)
BATCH_SIZE = 2_000_000

# Load vertices and faces initially
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

# Precompute normals for face-centric point addition
face_normals = compute_face_normals(vertices[:, :3], faces)
vertex_normals = compute_vertex_normals(vertices[:, :3], faces, face_normals)

# Setup cameras and targets
with torch.no_grad():
    R = Renderer(
        50, 256, dr.RasterizeGLContext(), "data/Colored_Bunny/1.obj", with_texture=True
    )
    targets = R.target_imgs


# Camera setup
cameras = [
    Camera(R.fov_x, R.fov_x, 256, 256, R.view_mats[i], R.mvps[i], R.camera_positions[i])
    for i in range(50)
]

pbar = tqdm(range(EPOCHS * 5 * 3))
best_loss = torch.inf

# Main training loop
for epoch in pbar:
    # Sample vertices and add new points based on face sampling
    sampled_points = sample_from_faces(
        vertices[..., :3],
        faces,
        face_normals.T,
        num_samples=v,
        normal_offset=0.01,
    )
    sampled_points = torch.cat(
        (sampled_points, torch.ones((sampled_points.shape[0], 1), device="cuda") * t),
        dim=-1,
    )
    all_points = torch.cat([vertices, sampled_points], dim=0).cuda()

    # Prepare for rendering
    optimizer.zero_grad()
    sh_coeffs, opacity, scaling, rotation = [], [], [], []

    idx = 0
    while idx < all_points.shape[0]:
        end_idx = min(idx + BATCH_SIZE, all_points.shape[0])
        sh, op, sc, rot = render_head(all_points[idx:end_idx].detach())
        sh_coeffs.append(sh)
        opacity.append(op)
        scaling.append(sc)
        rotation.append(rot)
        idx += BATCH_SIZE

    sh_coeffs = torch.cat(sh_coeffs, dim=0).cuda()
    opac = torch.cat(opacity, dim=0).cuda()
    scaling = torch.cat(scaling, dim=0).cuda()
    rotation = torch.cat(rotation, dim=0).cuda()

    # Random camera selection and rendering
    rand_cam = np.random.randint(0, len(cameras))
    camera = cameras[rand_cam]
    est = render_with_seperate_args(
        camera,
        all_points[..., :3],
        3,
        opac,
        scaling,
        rotation,
        sh_coeffs,
        torch.tensor([0.0, 0.0, 0.0]).cuda(),
    )["render"].permute(1, 2, 0)

    loss = torch.nn.functional.mse_loss(est, targets[rand_cam][..., :3]) + 0.01 * ssim(
        est, targets[rand_cam][..., :3]
    )
    loss.backward()
    optimizer.step()

    plt.imsave("test_2.png", est.cpu().detach().numpy().clip(0, 1))

    if loss < best_loss:
        best_loss = loss
        pbar.set_description(f"Epoch :{epoch} | Loss: {loss.item()}")
        plt.imsave("test_2_best.png", est.cpu().detach().numpy().clip(0, 1))


render_head.eval()

sh_coeffs, opacity, scaling, rotation = [], [], [], []
idx = 0
while idx < all_points.shape[0]:
    end_idx = min(idx + BATCH_SIZE, all_points.shape[0])
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


shs_view = sh_coeffs.transpose(1, 2).view(-1, 3, (3 + 1) ** 2)
dir_pp = vertices[..., :3] - R.camera_positions[0].repeat(sh_coeffs.shape[0], 1)
dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

plt.imsave(
    "test_test.png",
    R.render_coloured(
        vertices[..., :3],
        faces,
        vertex_normals,
        vertex_colors=colors_precomp,
        view_idx=0,
        albedo=1.0,
    )[0][..., :3]
    .cpu()
    .detach()
    .numpy()
    .clip(0, 1),
)

shs_view = sh_coeffs.transpose(1, 2).view(-1, 3, (3 + 1) ** 2)
dir_pp = vertices[..., :3] - R.camera_positions[-1].repeat(sh_coeffs.shape[0], 1)
dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

plt.imsave(
    "test_test2.png",
    R.render_coloured(
        vertices[..., :3],
        faces,
        vertex_normals,
        vertex_colors=colors_precomp,
        view_idx=-1,
        albedo=1.0,
    )[-1][..., :3]
    .cpu()
    .detach()
    .numpy()
    .clip(0, 1),
)

print("Training complete.")
