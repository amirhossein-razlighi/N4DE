import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from models_animated import SDFModule
from render import Renderer
from utils import *
from tqdm import tqdm

def gauss_kernel(size=5, device=torch.device("cuda"), channels=3):
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    kernel /= 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
    out = torch.nn.functional.conv2d(img.cuda(), kernel.cuda(), groups=img.shape[1])
    return out


def img_loss(imgs, target_imgs, multi_scale=True):
    loss = 0
    kernel = gauss_kernel()
    count = 0
    for i in range(imgs.shape[0]):
        count += 1
        loss = loss + (imgs[i] - target_imgs[i]).square().mean()
        if multi_scale:
            current_est = imgs[i].permute(2, 0, 1).unsqueeze(0)
            current_gt = target_imgs[i].permute(2, 0, 1).unsqueeze(0)
            for j in range(4):
                filtered_est = conv_gauss(current_est, kernel)
                filtered_gt = conv_gauss(current_gt, kernel)
                down_est = downsample(filtered_est)
                down_gt = downsample(filtered_gt)

                current_est = down_est
                current_gt = down_gt

                loss = loss + (current_est - current_gt).square().mean() / (j + 1)

    loss = loss / count
    return loss

"""
A loss function, doing this:
for (a,b) -> df/dt + <grad(f), F> = 0
for t=t_i -> f = g_i
"""
def loss_morphing(gt_sdf, pred_sdf, F, vertices, t):
    grad = gradient(pred_sdf, vertices)
    df_dt = grad[:, 3]
    df_3d = grad[:, :3]
    if t == int(t):
        return (gt_sdf - pred_sdf).abs().mean()
    else:
        return (gt_sdf - pred_sdf).abs().mean() + (df_dt + torch.sum(df_3d * F, dim=-1)).abs().mean()
    
def main(config):
    model_cfg = Namespace(
        dim=4, out_dim=1, hidden_size=512, n_blocks=4, z_dim=1, const=60.0
    )
    f = config.init_ckpt
    # f = None
    module = SDFModule(cfg=model_cfg, f=f, save_dir=config.expdir).cuda()
    logger = SummaryWriter(log_dir=config.expdir, flush_secs=5)

    optimizer = torch.optim.Adam(
        list(module.parameters()), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.num_frames is None:
        config.num_frames = 1

    qbar = tqdm(range(config.epochs))
    best_loss = np.inf

    gt_sdf = torch.zeros(config.max_v, 1).cuda()
    F = torch.zeros(config.max_v, 1).cuda()
    vertices = torch.zeros((config.max_v, 4)).cuda() # last column is time
    normals = torch.zeros((config.max_v, 4)).cuda()
    faces = torch.empty((config.max_v, 3), dtype=torch.int32).cuda()
    vertices.requires_grad_()

    for e in qbar:
        qbar.set_description(f"Epoch {e}")

        laplace_lam = config.max_laplace_lam

        if e >= config.fine_e:
            laplace_lam = config.min_laplace_lam
            mesh_res = config.mesh_res_limit + np.random.randint(low=-3, high=3)

        else:
            laplace_lam = config.max_laplace_lam
            mesh_res = config.mesh_res_base + np.random.randint(low=-3, high=3)

        images = []
        video_tensor = torch.zeros((config.num_frames, 3, config.res, config.res))
        # Iterating over frames (in one epoch)
        for t in range(config.num_frames):
            qbar.set_postfix_str(f"Frame: {t}")

            with torch.no_grad():
                name = f"{t+1}.ply"
                # Comment the code below, unless you run for Anim_1/
                # if t == 0:
                #     name = f"{t}.obj"
                # else:
                #     name = f"{t}.ply"

                R = Renderer(
                    config.num_views,
                    config.res,
                    fname=config.mesh + name,
                    scale=config.scale,
                )
                target_imgs = R.target_imgs

            logger.add_image(
                f"target_{t}", target_imgs[-1].permute(2, 0, 1).clamp(0, 1)
            )

            with torch.no_grad():
                vertices_np, faces_np = module.get_zero_points(mesh_res=mesh_res, t=t)
                v = vertices_np.shape[0]
                f = faces_np.shape[0]

                concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
                vertices.data[:v] = torch.from_numpy(concated)
                faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            vertices.grad = None

            edges = compute_edges(vertices[:v, :3], faces[:f])
            L = laplacian_simple(vertices[:v, :3], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v]).T @ vertices[:v]))

            face_normals = compute_face_normals(vertices[:v, :3], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v, :3], faces[:f], face_normals
            )
            imgs = R.render(vertices[:v, :3], faces[:f], vertex_normals)
            images.append(imgs[-1])
            # Computing E
            loss = img_loss(imgs, target_imgs, multi_scale=True)
            loss = loss + laplace_lam * laplacian_loss
            loss.backward()
            logger.add_scalar("image loss", loss.item(), global_step=e)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    module.state_dict(),
                    f"{config.expdir}/best.ckpt",
                )

            with torch.no_grad():
                dE_dx = vertices.grad[:v].detach()
            idx = 0
            while idx < v:
                optimizer.zero_grad()
                min_i = idx
                max_i = min(min_i + config.batch_size, v)
                vertices_subset = vertices[min_i:max_i]
                vertices_subset.requires_grad_()
                pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
                normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()

                # Flow field (F) = sum of (dx/dt * dE/dx)
                F[min_i:max_i] = torch.nan_to_num(
                    torch.sum(
                        normals[min_i:max_i] * dE_dx[min_i:max_i], dim=-1, keepdim=True
                    )
                )
                logger.add_scalar("flow field magnitude", F[min_i:max_i].norm().item(), global_step=e)
                # Ground truth SDF = predicted SDF + epsilon * Flow field
                gt_sdf[min_i:max_i] = (pred_sdf + config.eps * F[min_i:max_i]).detach()
                idx += config.batch_size
            n_batches = v // config.batch_size
            if n_batches == 0:
                n_batches = 1
            for j in range(config.iters):
                optimizer.zero_grad()
                idx = 0
                while idx < v:
                    min_i = idx
                    max_i = min(min_i + config.batch_size, v)
                    vertices_subset = vertices[min_i:max_i].detach()
                    pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
                    
                    # d_Phi / d_t
                    loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches
                    
                    # term for morphing
                    # d_phi/dt * <grad(phi), F>
                    # loss += normals[min_i:max_i, 3].abs().mean() * torch.sum(normals[min_i:max_i, :3] * F[min_i:max_i], dim=-1).abs().mean() / n_batches
                    
                    # Force F to have small magnitude
                    # loss += F[min_i:max_i].abs().mean() / n_batches

                    loss.backward()
                    logger.add_scalar("loss", loss.item(), global_step=e)
                    idx += config.batch_size
                # update the parameters
                optimizer.step()

        if e % config.video_log_freq == 0:
            images = torch.stack(images)
            video_tensor = images.permute(0, 3, 1, 2).unsqueeze(0)
            logger.add_video(
                "video",
                video_tensor,
                global_step=(e),
                fps=10,
            )
        if e % config.img_log_freq == 0:
            grid = make_grid(images[0].permute(2, 0, 1).clamp(0, 1))
            for img in images[1:]:
                grid = torch.cat((grid, make_grid(img.permute(2, 0, 1).clamp(0, 1))), dim=2)
            logger.add_image(
                "est",
                grid,
                global_step=(e),
            )
        if e % config.mesh_log_freq == 0:
            with torch.no_grad():
                mse = (imgs - target_imgs).square().mean()
                psnr = -10.0 * torch.log10(mse)
                logger.add_scalar("psnr", psnr, global_step=(e))
            mesh = trimesh.Trimesh(vertices_np, faces_np)
            cd = compute_trimesh_chamfer(R.mesh, mesh)
            logger.add_scalar("cd", cd, global_step=(e))
            mesh.export(f"{config.expdir}/mesh_{(e):07d}.ply")
        if e % config.ckpt_log_freq == 0:
            torch.save(
                module.state_dict(),
                f"{config.expdir}/iter_{(e):07d}.ckpt",
            )



def old_iteration():
    for t in range(config.num_frames):
        print(f"Frame: {t}")
        gt_sdf = torch.zeros(config.max_v, 1).cuda()
        F = torch.zeros(config.max_v, 1).cuda()
        vertices = torch.zeros((config.max_v, 3)).cuda()
        normals = torch.zeros((config.max_v, 3)).cuda()
        faces = torch.empty((config.max_v, 3), dtype=torch.int32).cuda()
        vertices.requires_grad_()
        with torch.no_grad():
            # name = f"{t}.obj"
            # Comment this unless you run for Anim_1/
            if t == 0:
                name = f"{t}.obj"
            else:
                name = f"{t}.ply"
            R = Renderer(
                config.num_views,
                config.res,
                fname=config.mesh + name,
                scale=config.scale,
            )
            target_imgs = R.target_imgs
        logger.add_image(f"target_{t}", target_imgs[-1].permute(2, 0, 1).clamp(0, 1))
        for e in range(config.epochs):
            laplace_lam = config.max_laplace_lam
            if e >= config.fine_e:
                laplace_lam = config.min_laplace_lam
                mesh_res = config.mesh_res_limit + np.random.randint(low=-3, high=3)
            else:
                laplace_lam = config.max_laplace_lam
                mesh_res = config.mesh_res_base + np.random.randint(low=-3, high=3)
            with torch.no_grad():
                vertices_np, faces_np = module.get_zero_points(mesh_res=mesh_res, t=t)
                v = vertices_np.shape[0]
                f = faces_np.shape[0]
                vertices.data[:v] = torch.from_numpy(vertices_np)
                faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))
            vertices.grad = None
            edges = compute_edges(vertices[:v], faces[:f])
            L = laplacian_simple(vertices[:v], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v]).T @ vertices[:v]))
            face_normals = compute_face_normals(vertices[:v], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v], faces[:f], face_normals
            )
            imgs = R.render(vertices[:v], faces[:f], vertex_normals)
            loss = img_loss(imgs, target_imgs, multi_scale=True)
            loss = loss + laplace_lam * laplacian_loss
            loss.backward()
            logger.add_scalar("loss", loss.item(), global_step=e)
            with torch.no_grad():
                dE_dx = vertices.grad[:v].detach()
            idx = 0
            while idx < v:
                optimizer.zero_grad()
                min_i = idx
                max_i = min(min_i + config.batch_size, v)
                vertices_subset = vertices[min_i:max_i]
                # add time parameter to the 3d inputs (x, y, z, t). the vertices are [n, x, y, z] so
                # we should add an axis to the end of the tensor.
                inp = torch.cat(
                    (
                        vertices_subset,
                        torch.ones((vertices_subset.shape[0], 1)).cuda() * t,
                    ),
                    dim=1,
                )
                vertices_subset.requires_grad_()
                pred_sdf = module.forward(inp.unsqueeze(0)).squeeze(0)
                normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()
                F[min_i:max_i] = torch.nan_to_num(
                    torch.sum(
                        normals[min_i:max_i] * dE_dx[min_i:max_i], dim=-1, keepdim=True
                    )
                )
                gt_sdf[min_i:max_i] = (pred_sdf + config.eps * F[min_i:max_i]).detach()
                idx += config.batch_size
            n_batches = v // config.batch_size
            if n_batches == 0:
                n_batches = 1
            for j in range(config.iters):
                optimizer.zero_grad()
                idx = 0
                while idx < v:
                    min_i = idx
                    max_i = min(min_i + config.batch_size, v)
                    vertices_subset = vertices[min_i:max_i].detach()
                    # add time parameter to the 3d inputs (x, y, z, t). the vertices are [n, x, y, z] so
                    # we should add an axis to the end of the tensor.
                    vertices_subset = torch.cat(
                        (
                            vertices_subset,
                            torch.ones((vertices_subset.shape[0], 1)).cuda() * t,
                        ),
                        dim=1,
                    )
                    pred_sdf = module.forward(vertices_subset.unsqueeze(0)).squeeze(0)
                    loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches
                    loss.backward()
                    idx += config.batch_size
                optimizer.step()
            if e % 1 == 0:
                print(f"Epoch: {(e + config.epochs * t)}")
            if e % config.img_log_freq == 0:
                logger.add_image(
                    "est",
                    imgs[-1].permute(2, 0, 1).clamp(0, 1),
                    global_step=(e + config.epochs * t),
                )
            if e % config.mesh_log_freq == 0:
                with torch.no_grad():
                    mse = (imgs - target_imgs).square().mean()
                    psnr = -10.0 * torch.log10(mse)
                    logger.add_scalar("psnr", psnr, global_step=(e + config.epochs * t))
                mesh = trimesh.Trimesh(vertices_np, faces_np)
                cd = compute_trimesh_chamfer(R.mesh, mesh)
                logger.add_scalar("cd", cd, global_step=(e + config.epochs * t))
                mesh.export(f"{config.expdir}/mesh_{(e + config.epochs * t):07d}.ply")
            if e % config.ckpt_log_freq == 0:
                torch.save(
                    module.state_dict(),
                    f"{config.expdir}/iter_{(e + config.epochs * t):07d}.ckpt",
                )


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    config = parse_config(create_dir=True, consider_max_dir=True)
    main(config)
