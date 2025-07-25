import yaml
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.fftpack
import time
import plyfile
import skimage.measure
from scipy.spatial import cKDTree as KDTree
import trimesh
import torch.nn as nn
from torch.nn import functional as F
import argparse
from render_utils import C0, C1, C2, C3, C4
from math import exp


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    return grad


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def dir_counter(LOGDIR, endswith=None, consider_max=False):
    if endswith is not None:
        if consider_max:
            length = [
                int(name.split("_")[0])
                for name in os.listdir(LOGDIR)
                if endswith in name
            ]
            if len(length) == 0:
                length = 1
            else:
                length = max(length) + 1
            return length
        else:
            return len([name for name in os.listdir(LOGDIR) if name.endswith(endswith)])
    if consider_max:
        length = [int(name.split("_")[0]) for name in os.listdir(LOGDIR)]
        if len(length) == 0:
            length = 1
        else:
            length = max(length) + 1
        return length
    return len([name for name in os.listdir(LOGDIR)])


def dict2namespace(config):
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_config(suffix="", create_dir=True, consider_max_dir=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file")
    parser.add_argument("--test_mesh", type=str, help="Test Mesh", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    if config.folder_name is not None and config._continue:
        config.expdir = os.path.join(config.logdir, config.folder_name)
    else:
        dir_count = (
            str(
                dir_counter(
                    config.logdir, endswith=config.exp, consider_max=consider_max_dir
                )
            )
            + "_"
            + config.exp
            + suffix
        )
        config.expdir = config.logdir + dir_count
    print("Experiment dir: ", config.expdir)
    print()
    config.test_mesh = args.test_mesh
    if create_dir:
        os.makedirs(config.expdir, exist_ok=True)

    return config


def compute_trimesh_chamfer(gt_mesh, gen_mesh, num_mesh_samples=100000):
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    gt_points_np = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_edges(vertices, faces, return_faces=False):
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)
    e12 = torch.cat([v1, v2], dim=1)
    e20 = torch.cat([v2, v0], dim=1)
    edges = torch.cat([e12, e20, e01], dim=0).long()
    edges, _ = edges.sort(dim=1)
    V = vertices.shape[0]
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(
        edges_hash.shape[0], dtype=torch.bool, device=vertices.device
    )
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    unique_idx = sort_idx[unique_mask]
    edges = torch.stack([u // V, u % V], dim=1)

    if return_faces:
        faces = inverse_idxs.reshape(3, faces.shape[0]).t()
        return edges.long(), faces.long()
    return edges.long()


def laplacian_simple(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    V = verts.shape[0]
    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse_coo_tensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, -deg0 * 0 - 1, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, -deg1 * 0 - 1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse_coo_tensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = (
        torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device) * deg[idx[0]]
    )
    L += torch.sparse_coo_tensor(idx, ones, (V, V))

    return L


def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse


def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3


def massmatrix_voronoi(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    http://www.alecjacobson.com/weblog/?p=863

    params
    ------

    v: vertex positions
    f: triangle indices
    """
    # Compute edge lengths
    l0 = (verts[faces[:, 1]] - verts[faces[:, 2]]).norm(dim=1)
    l1 = (verts[faces[:, 2]] - verts[faces[:, 0]]).norm(dim=1)
    l2 = (verts[faces[:, 0]] - verts[faces[:, 1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)

    # Compute cosines of the corners of the triangles
    cos0 = (l[:, 1].square() + l[:, 2].square() - l[:, 0].square()) / (
        2 * l[:, 1] * l[:, 2]
    )
    cos1 = (l[:, 2].square() + l[:, 0].square() - l[:, 1].square()) / (
        2 * l[:, 2] * l[:, 0]
    )
    cos2 = (l[:, 0].square() + l[:, 1].square() - l[:, 2].square()) / (
        2 * l[:, 0] * l[:, 1]
    )
    cosines = torch.stack((cos0, cos1, cos2), dim=1)

    # Convert to barycentric coordinates
    barycentric = cosines * l
    barycentric = barycentric / torch.sum(barycentric, dim=1)[..., None]

    # Compute areas of the faces using Heron's formula
    areas = (
        0.25
        * ((l0 + l1 + l2) * (l0 + l1 - l2) * (l0 - l1 + l2) * (-l0 + l1 + l2)).sqrt()
    )

    # Compute the areas of the sub triangles
    tri_areas = areas[..., None] * barycentric

    # Compute the area of the quad
    cell0 = 0.5 * (tri_areas[:, 1] + tri_areas[:, 2])
    cell1 = 0.5 * (tri_areas[:, 2] + tri_areas[:, 0])
    cell2 = 0.5 * (tri_areas[:, 0] + tri_areas[:, 1])
    cells = torch.stack((cell0, cell1, cell2), dim=1)

    # Different formulation for obtuse triangles
    # See http://www.alecjacobson.com/weblog/?p=874
    cells[:, 0] = torch.where(cosines[:, 0] < 0, 0.5 * areas, cells[:, 0])
    cells[:, 1] = torch.where(cosines[:, 0] < 0, 0.25 * areas, cells[:, 1])
    cells[:, 2] = torch.where(cosines[:, 0] < 0, 0.25 * areas, cells[:, 2])

    cells[:, 0] = torch.where(cosines[:, 1] < 0, 0.25 * areas, cells[:, 0])
    cells[:, 1] = torch.where(cosines[:, 1] < 0, 0.5 * areas, cells[:, 1])
    cells[:, 2] = torch.where(cosines[:, 1] < 0, 0.25 * areas, cells[:, 2])

    cells[:, 0] = torch.where(cosines[:, 2] < 0, 0.25 * areas, cells[:, 0])
    cells[:, 1] = torch.where(cosines[:, 2] < 0, 0.25 * areas, cells[:, 1])
    cells[:, 2] = torch.where(cosines[:, 2] < 0, 0.5 * areas, cells[:, 2])

    # Sum the quad areas to get the voronoi cell
    return torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)


def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2]),
    ]

    c = torch.linalg.cross(v[1] - v[0], v[2] - v[0], dim=0)
    n = c / torch.norm(c, dim=0)
    return n


def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))


def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2]),
    ]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0 * d1, 0)
        face_angle = safe_acos(torch.sum(d0 * d1, 0))
        nn = face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)


from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
# ----------------------------------------------------------------------------


def projection(x=0.1, n=1.0, f=50.0):
    return np.array(
        [
            [n / x, 0, 0, 0],
            [0, n / -x, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    ).astype(np.float32)


def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]).astype(
        np.float32
    )


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]).astype(
        np.float32
    )


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]).astype(
        np.float32
    )


def random_rotation():
    r = R.random().as_matrix()
    r = np.hstack([r, np.zeros((3, 1))])
    r = np.vstack([r, np.zeros((1, 4))])
    r[3, 3] = 1
    return r


def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m = np.identity(3)
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode="constant")
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m


# ----------------------------------------------------------------------------
# Bilinear downsample by 2x.
# ----------------------------------------------------------------------------


def bilinear_downsample(x):
    w = (
        torch.tensor(
            [[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]],
            dtype=torch.float32,
            device=x.device,
        )
        / 64.0
    )
    w = w.expand(x.shape[-1], 1, 4, 4)
    x = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1]
    )
    return x.permute(0, 2, 3, 1)


def save_image(fn, x):
    import imageio

    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)


def plot_grad_flow(named_parameters, writer, epoch):
    """Logs the gradients flowing through different layers in the net during training to TensorBoard.
    Can be used for checking for possible gradient vanishing / exploding problems.

    "plot_grad_flow(self.model.named_parameters(), self.writer, epoch)" to visualize the gradient flow in TensorBoard,
    where `self.writer` is an instance of `SummaryWriter` and `epoch` is the current epoch number.
    """
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is not None and p.grad.nelement() > 0:
                writer.add_histogram("layers_grad/" + n, p.grad, epoch)


def evaluate_mesh_at_specific_time(model, time, device, mesh_res=128, offset=0):
    bound = 1.0
    batch_size = 20000
    xs, ys, zs = np.meshgrid(
        np.arange(mesh_res), np.arange(mesh_res), np.arange(mesh_res)
    )
    grid = np.concatenate(
        [ys[..., np.newaxis], xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1
    ).astype(np.float32)

    grid = (grid / (mesh_res - 1) - 0.5) * 2.0 * bound
    grid = grid.reshape(-1, 3)

    voxel_size = 2.0 * bound / (mesh_res - 1)
    voxel_origin = -1.0 * bound

    dists_list = []
    for i in range(grid.shape[0], batch_size):
        start_idx, end_idx = i, min(i + batch_size, grid.shape[0])
        points = grid[start_idx:end_idx]

        with torch.no_grad():
            xyz = torch.from_numpy(points).to(device)
            t = torch.ones(xyz.shape[0], 1).to(device) * time
            xyz = torch.cat([xyz, t], dim=1)
            dists = model(xyz)
            dists = dists.cpu().numpy()

        dists_list.append(dists)

    dists = np.concatenate([x.reshape(-1, 1) for x in dists_list], axis=0).reshape(-1)
    field = dists.reshape(mesh_res, mesh_res, mesh_res)

    verts, faces, _, _ = skimage.measure.marching_cubes_lewiner(
        field,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts += voxel_origin
    verts -= offset
    return verts, faces


def get_vol_points(vertices, grad, F, num_vol_points, sigma=None, both=None):
    points_per_vert = 1

    if sigma is not None:
        a = sigma
    else:
        a = 0.1

    rand_d = torch.randn(vertices.shape[0], points_per_vert, 1).cuda() * a
    vol_points = vertices.unsqueeze(1) + rand_d * grad.unsqueeze(1)
    F_ext = F.unsqueeze(1).repeat(1, points_per_vert, 1)
    vol_points = vol_points.view(-1, 3)
    F_ext = F_ext.view(-1, 1)
    idx = np.random.choice(vertices.shape[0] * points_per_vert, num_vol_points // 2)
    near_points = vol_points.view(-1, 3)[idx]
    near_F = F_ext.view(-1, 1)[idx]

    if both:
        rand_d = torch.randn(vertices.shape[0], points_per_vert, 1).cuda() * 0.3
    else:
        rand_d = torch.randn(vertices.shape[0], points_per_vert, 1).cuda() * a

    vol_points = vertices.unsqueeze(1) + rand_d * grad.unsqueeze(1)
    F_ext = F.unsqueeze(1).repeat(1, points_per_vert, 1)
    vol_points = vol_points.view(-1, 3)
    F_ext = F_ext.view(-1, 1)
    idx = np.random.choice(vertices.shape[0] * points_per_vert, num_vol_points // 2)
    far_points = vol_points.view(-1, 3)[idx]
    far_F = F_ext.view(-1, 1)[idx]

    return torch.cat([near_points, far_points], dim=0), torch.cat(
        [near_F, far_F], dim=0
    )


freq_bands = None


def positional_encoding(inputs, out_dim=512, device="cuda:0"):
    """
    Do positional encoding in each dimension
    and return a vector in device of size (N, num_input_dims * out_dim) | i.e. (N, 3 * 64)
    """
    global freq_bands

    if freq_bands is None:
        freq_bands = torch.pow(
            10000, -torch.arange(0, out_dim, 2, device=device).float() / out_dim
        )

    batch_size, num_dims = inputs.shape
    pe = torch.zeros(batch_size, num_dims * out_dim, device=device)

    # Expand inputs and frequencies for broadcasting
    inputs_expanded = inputs.unsqueeze(-1)  # (batch_size, num_dims, 1)
    freq_expanded = freq_bands.unsqueeze(0).unsqueeze(0)  # (1, 1, out_dim/2)

    # Compute arguments for sin and cos
    args = inputs_expanded * freq_expanded  # (batch_size, num_dims, out_dim/2)

    # Compute sin and cos
    sin_vals = torch.sin(args)
    cos_vals = torch.cos(args)

    # Interleave sin and cos values
    pe_vals = torch.stack([sin_vals, cos_vals], dim=-1).flatten(start_dim=-2)

    # Reshape to desired output shape
    pe = pe_vals.reshape(batch_size, -1)

    return pe


def calculate_view_direction(vertices, cam_positions):
    """
    Calculate the view direction from the camera positions to the vertices
    """
    cam_positions = cam_positions.unsqueeze(1)
    view_direction = vertices - cam_positions
    view_direction = view_direction / torch.norm(view_direction, dim=2, keepdim=True)
    return view_direction


def load_K_Rt_from_P(filename, P=None):
    import cv2 as cv

    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_damping_factor(
    epoch, initial_damping_factor, decay_rate, decay_type="exponential"
):
    if decay_type == "linear":
        return initial_damping_factor * (1 - decay_rate * epoch)
    elif decay_type == "exponential":
        return initial_damping_factor * (decay_rate**epoch)
    else:
        raise ValueError("Unsupported decay_type. Use 'linear' or 'exponential'.")


def calculate_volume(vertices, faces):
    """
    Calculate the inside volume of a mesh.

    Args:
    vertices (np.ndarray): Array of shape (N, 3) representing the vertices of the mesh.
    faces (np.ndarray): Array of shape (M, 3) representing the faces of the mesh.

    Returns:
    float: The inside volume of the mesh.
    """
    volume = 0.0

    for face in faces:
        v0, v1, v2 = vertices[face]

        # Calculate the signed volume of the tetrahedron formed by the face and the origin | (A x B) . C  / 6
        tetra_volume = torch.dot(torch.cross(v0, v1), v2) / 6.0

        volume += tetra_volume

    return torch.abs(volume)


def inverse_sigmoid(x):
    x = x.clamp(min=1e-7, max=1 - 1e-7)  # to avoid log(0) and log(inf)
    return torch.log(x / (1 - x))


def report_average_metrics(metrics_dict, exp_directory):
    clone_dict = {}
    for key, val in metrics_dict.items():
        if isinstance(val, torch.Tensor):
            clone_dict[key] = val.mean().item()
        elif isinstance(val, list):
            clone_dict[key] = np.mean(val)

    with open(os.path.join(exp_directory, "metrics.txt"), "a") as f:
        f.write(str(clone_dict) + "\n")


def sh_l2_regularization(sh_coeffs):
    return torch.mean(sh_coeffs**2)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def render_colors_from_sh(sh_coeffs, directions, sh_order):
    """
    Render colors from SH coefficients.

    Parameters:
    - sh_coeffs: torch.Tensor, the SH coefficients, shape (N, num_sh_coeffs).
    - directions: torch.Tensor, the viewing directions, shape (N, 3).
    - sh_order: int, the order of spherical harmonics.

    Returns:
    - colors: torch.Tensor, the rendered colors, shape (N, 3).
    """
    evaled_sh_colors = eval_sh(sh_order, sh_coeffs, directions)
    rgb_colors = SH2RGB(evaled_sh_colors)
    return rgb_colors


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


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


def save_epoch_number_to_a_file(epoch_number, file_path):
    with open(file_path, "w") as f:
        f.write(str(epoch_number))


def load_epoch_number_from_a_file(file_path):
    with open(file_path, "r") as f:
        return int(f.read())
