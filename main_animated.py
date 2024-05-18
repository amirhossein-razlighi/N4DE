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
from loss import *


def main(config):
    device = torch.device("cuda:0")
    model_cfg = Namespace(
        dim=4,
        out_dim=1,
        hidden_size=512,
        n_blocks=4,
        z_dim=1,
        const=60.0,
    )
    f = config.init_ckpt
    model = SDFModule(cfg=model_cfg, f=f, save_dir=config.expdir, device=device).to(
        device
    )
    logger = SummaryWriter(log_dir=config.expdir, flush_secs=5)

    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.num_frames is None:
        config.num_frames = 1

    if config._continue:
        qbar = tqdm(range(config.last_epoch, config.epochs))
    else:
        qbar = tqdm(range(config.epochs))
    best_loss = np.inf

    gt_sdf = torch.zeros(config.max_v, 1).to(device)
    F = torch.zeros(config.max_v, 1).to(device)
    vertices = torch.zeros((config.max_v, 4)).to(device)  # last element is time
    normals = torch.zeros((config.max_v, 4)).to(device)
    faces = torch.empty((config.max_v, 3), dtype=torch.int32).to(device)
    vertices.requires_grad_()

    # renderers = []
    # print("Rendering target images")
    # for t in range(config.num_frames):
    #     with torch.no_grad():
    #         # Comment the code below, unless you run for Anim_1/
    #         # if t == 0:
    #         #     name = f"{t}.obj"
    #         # else:
    #         #     name = f"{t}.ply"
    #         name = f"{t + 1}.ply"
    #         R = Renderer(
    #             config.num_views,
    #             config.res,
    #             fname=config.mesh + name,
    #             scale=config.scale,
    #             device=device,
    #         )
    #         renderers.append(R)
    # print("Rendering done")

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
        depths = []
        video_tensor = torch.zeros((config.num_frames, 3, config.res, config.res))
        # Iterating over frames (in one epoch)
        for t in range(config.num_frames):
            qbar.set_postfix_str(f"Frame: {t}")
            # R = renderers[t]

            with torch.no_grad():
                name = f"{t + 1}.ply"
                R = Renderer(
                    config.num_views,
                    config.res,
                    fname=config.mesh + name,
                    scale=config.scale,
                    device=device,
                )
            target_imgs = R.target_imgs
            target_depths = target_imgs[..., 3]

            t = t / 10
            logger.add_image(
                f"target_{t}", target_imgs[-1][..., :3].permute(2, 0, 1).clamp(0, 1)
            )
            logger.add_image(f"target_depth_{t}", target_depths[-1].unsqueeze(0))

            with torch.no_grad():
                vertices_np, faces_np = model.get_zero_points(
                    mesh_res=mesh_res, t=t, device=device
                )
                v = vertices_np.shape[0]
                f = faces_np.shape[0]

                concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
                vertices.data[:v] = torch.from_numpy(concated)
                faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            vertices.grad = None

            edges = compute_edges(vertices[:v, :3], faces[:f])
            L = laplacian_simple(vertices[:v, :3], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v, :3]).T @ vertices[:v, :3]))

            face_normals = compute_face_normals(vertices[:v, :3], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v, :3], faces[:f], face_normals
            )
            imgs = R.render(vertices[:v, :3], faces[:f], vertex_normals, device=device)
            images.append(imgs[-1][..., :3])
            depths.append(imgs[-1][..., 3])
            # Computing E
            loss = img_loss(
                imgs,
                target_imgs,
                multi_scale=True,
                device=device,
                config=config,
                include_depth=config.include_depth,
            )
            loss = loss + laplace_lam * laplacian_loss
            loss.backward()
            logger.add_scalar("image loss", loss.item(), global_step=e)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    model.state_dict(),
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
                pred_sdf = model(vertices_subset.unsqueeze(0)).squeeze(0)
                normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()

                # Flow field (F) = sum of (dx/dt * dE/dx)
                F[min_i:max_i] = torch.nan_to_num(
                    torch.sum(
                        normals[min_i:max_i] * dE_dx[min_i:max_i], dim=-1, keepdim=True
                    )
                )
                logger.add_scalar(
                    "flow field magnitude",
                    F[min_i:max_i].norm().item(),
                    global_step=e,
                )
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
                    pred_sdf = model(vertices_subset.unsqueeze(0)).squeeze(0)

                    # d_Phi / d_t
                    loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches

                    # d phi/ dt time smoothing
                    # loss += config.time_smoothing * (
                    #     time_smoothing_loss(normals[min_i:max_i, 3]) / n_batches
                    # )
                    #TODO: Experimenting this idea ...
                    loss += config.time_smoothing * (
                        time_smoothing_inter_frames_loss(
                            t, t + 0.1, model, vertices[min_i:max_i]
                        )
                        / n_batches
                    )

                    loss += (
                        config.eikonal_coeff
                        * eikonal_loss(normals[min_i:max_i, :3])
                        / n_batches
                    )

                    # The term for gradients to be orthogonal to vector field F
                    loss += (
                        config.orthogonal_coeff
                        * orthogonal_normals(normals[min_i:max_i], F[min_i:max_i])
                        / n_batches
                    )

                    # CHECK: input vertices should be between -1 and 1

                    # t between 0 and 1 -> 0 and 0.1 and 0.2
                    # for intermediate t between 0 and 0.1

                    loss.backward()
                    logger.add_scalar("loss", loss.item(), global_step=e)
                    idx += config.batch_size
                # update the parameters
                optimizer.step()

        if e % config.video_log_freq == 0:
            images_ = torch.stack(images)
            video_tensor = images_.permute(0, 3, 1, 2).unsqueeze(0)
            logger.add_video(
                "video",
                video_tensor,
                global_step=(e),
                fps=1,
            )
        if e % config.img_log_freq == 0:
            est_grid = make_grid(images[0][..., :3].clamp(0, 1))
            depth_grid = make_grid(depths[0].unsqueeze(0).clamp(0, 1))
            for img in images[1:]:
                est_grid = torch.cat(
                    (est_grid, make_grid(img[..., :3].clamp(0, 1))),
                    dim=1,
                )
            for depth in depths[1:]:
                depth_grid = torch.cat(
                    (depth_grid, make_grid(depth.unsqueeze(0).clamp(0, 1))),
                    dim=2,
                )
            est_grid = est_grid.permute(2, 0, 1)  # reshape into (C, H, W)
            logger.add_image(
                "est",
                est_grid,
                global_step=(e),
            )
            if config.include_depth:
                logger.add_image(
                    "est_depth",
                    depth_grid,
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
                model.state_dict(),
                f"{config.expdir}/iter_{(e):07d}.ckpt",
            )


def old_iteration():
    for t in range(config.num_frames):
        print(f"Frame: {t}")
        gt_sdf = torch.zeros(config.max_v, 1).to(device)
        F = torch.zeros(config.max_v, 1).to(device)
        vertices = torch.zeros((config.max_v, 3)).to(device)
        normals = torch.zeros((config.max_v, 3)).to(device)
        faces = torch.empty((config.max_v, 3), dtype=torch.int32).to(device)
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
                vertices_np, faces_np = model.get_zero_points(
                    mesh_res=mesh_res, t=t, device=device
                )
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
            imgs = R.render(vertices[:v], faces[:f], vertex_normals, device=device)
            loss = img_loss(imgs, target_imgs, multi_scale=True, device=device)
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
                        torch.ones((vertices_subset.shape[0], 1)).to(device) * t,
                    ),
                    dim=1,
                )
                vertices_subset.requires_grad_()
                pred_sdf = model(inp.unsqueeze(0)).squeeze(0)
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
                            torch.ones((vertices_subset.shape[0], 1)).to(device) * t,
                        ),
                        dim=1,
                    )
                    pred_sdf = model(vertices_subset.unsqueeze(0)).squeeze(0)
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
                    model.state_dict(),
                    f"{config.expdir}/iter_{(e + config.epochs * t):07d}.ckpt",
                )


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    config = parse_config(create_dir=True, consider_max_dir=True)
    main(config)
