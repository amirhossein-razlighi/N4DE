import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from render import Renderer
from utils import *
from tqdm import tqdm
from loss import *
import matplotlib.pyplot as plt
from accelerate import Accelerator


def perform_training_distributed(
    config,
    device,
    model,
    logger,
    config_dict_for_mesh_logging,
    ms_ssim,
    optimizer,
    lr_scheduler,
    qbar,
    best_loss,
    gt_sdf,
    F,
    vertices,
    normals,
    faces,
    prof_,  # torch profiler
):
    use_amp = config.use_amp
    accelerator = Accelerator(mixed_precision="fp16" if use_amp else None)
    device = accelerator.device
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    for e in qbar:
        qbar.set_description(f"Epoch {e}")
        prof_.step()

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
        avg_dict = {
            "photometric_loss": 0,
            "sdf_l1_loss": 0,
            "sdf_loss": 0,
            "cd_loss": 0,
            # "mape_loss": 0,
            "laplace_loss": 0,
            "mssim_loss": 0,
            "eikonal_loss": 0,
            "image_mse_loss": 0,
            "psnr": 0,
        }

        # Iterating over frames (in one epoch)
        for t in range(config.num_frames):
            qbar.set_postfix(frame=t)

            with torch.no_grad():
                # name = f"{t + 1}.ply"
                name = f"{t + 1}.obj"
                R = Renderer(
                    config.num_views,
                    config.res,
                    fname=config.mesh + name,
                    scale=config.scale,
                    device="cuda:0",
                )
            target_imgs = R.target_imgs
            target_depths = target_imgs[..., 3]
            accelerator.wait_for_everyone()  # wait for rendering to finish

            # t = t * or / 10
            logger.add_image(
                f"targets/target_{t}",
                target_imgs[-1][..., :3].permute(2, 0, 1).clamp(0, 1),
            )
            logger.add_image(
                f"targets/target_depth_{t}", target_depths[-1].unsqueeze(0)
            )

            with torch.no_grad():
                vertices_np, faces_np = model.get_zero_points(
                    mesh_res=mesh_res,
                    t=t,
                    device=device,
                    batch_size=config.batch_size_for_mc,
                )
                v = vertices_np.shape[0]
                f = faces_np.shape[0]

                concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
                vertices.data[:v] = torch.from_numpy(concated)
                faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            if e % config.tensorboard_mesh_log_freq == 0:
                logger.add_mesh(
                    f"mesh_{t}",
                    vertices=vertices[:v, :3].unsqueeze(0).cpu().detach().numpy(),
                    faces=faces[:f].unsqueeze(0).cpu().detach().numpy(),
                    config_dict=config_dict_for_mesh_logging,
                    global_step=e,
                )
            vertices.grad = None

            edges = compute_edges(vertices[:v, :3], faces[:f])
            L = laplacian_simple(vertices[:v, :3], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v, :3]).T @ vertices[:v, :3]))

            face_normals = compute_face_normals(vertices[:v, :3], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v, :3], faces[:f], face_normals
            )
            imgs = R.render(
                vertices[:v, :3], faces[:f], vertex_normals, device="cuda:0"
            )
            accelerator.wait_for_everyone()  # wait for rendering to finish
            images.append(imgs[-1][..., :3])
            depths.append(imgs[-1][..., 3])

            with accelerator.autocast(enabled=use_amp):
                # Computing E
                loss = img_loss(
                    imgs,
                    target_imgs,
                    multi_scale=True,
                    device=device,
                    config=config,
                    include_depth=config.include_depth,
                )
                avg_dict["image_mse_loss"] += loss.item()
                avg_dict["psnr"] += torch.functional.F.mse_loss(
                    imgs[..., :3], target_imgs[..., :3], reduction="mean"
                )
                loss = loss + laplace_lam * laplacian_loss
                if config.ms_ssim_lam > 0:
                    ms_ssim_loss = 1 - ms_ssim(
                        imgs[..., :3].permute(0, 3, 1, 2),
                        target_imgs[..., :3].permute(0, 3, 1, 2),
                    )
                    loss = (
                        1 - config.ms_ssim_lam
                    ) * loss + config.ms_ssim_lam * torch.max(
                        ms_ssim_loss, torch.tensor(0.0)
                    )
                accelerator.backward(loss)
            avg_dict["photometric_loss"] += loss.item()
            avg_dict["laplace_loss"] += laplacian_loss.item()
            if config.ms_ssim_lam > 0:
                avg_dict["mssim_loss"] += ms_ssim_loss.item()

            plt.imsave(
                f"{config.expdir}/img_{t}.png",
                imgs[-1][..., :3].cpu().detach().numpy().clip(0, 1),
            )
            plt.imsave(
                f"{config.expdir}/target_img_{t}.png",
                target_imgs[-1].cpu().detach().numpy().clip(0, 1),
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                accelerator.wait_for_everyone()
                os.makedirs(f"{config.expdir}/best", exist_ok=True)
                accelerator.save_model(
                    model,
                    f"{config.expdir}/best/model.pth",
                )
                accelerator.save_model(
                    optimizer,
                    f"{config.expdir}/best/optim.pth",
                )
                accelerator.save_model(
                    lr_scheduler["warmup_scheduler"],
                    f"{config.expdir}/best/warmup_sched.pth",
                )
                accelerator.save_model(
                    lr_scheduler["scheduler"],
                    f"{config.expdir}/best/lr_sched.pth",
                )

            with torch.no_grad():
                dE_dx = vertices.grad[
                    :v
                ].detach()  # NOTE: dE / dx = - dx / dt (refer to the paper)
            with accelerator.autocast(enabled=use_amp):
                idx = 0
                while idx < v:
                    optimizer.zero_grad()
                    min_i = idx
                    max_i = min(min_i + config.batch_size, v)
                    vertices_subset = vertices[min_i:max_i]
                    vertices_subset.requires_grad_()
                    with torch.autocast(
                        device_type=device, dtype=torch.float16, enabled=use_amp
                    ):
                        pred_sdf = model(vertices_subset)
                    normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()

                    # Flow field (F) = sum of (d phi/dx * - dx / dt)
                    F[min_i:max_i] = torch.nan_to_num(
                        torch.sum(
                            normals[min_i:max_i] * dE_dx[min_i:max_i],
                            dim=-1,
                            keepdim=True,
                        )
                    )
                    logger.add_scalar(
                        "flow field magnitude",
                        F[min_i:max_i].norm().item(),
                        global_step=e,
                    )
                    # Ground truth SDF = predicted SDF + epsilon * Flow field
                    gt_sdf[min_i:max_i] = (
                        pred_sdf + config.eps * F[min_i:max_i]
                    ).detach()
                    idx += config.batch_size
            n_batches = v // config.batch_size
            if n_batches == 0:
                n_batches = 1
            with accelerator.autocast(enabled=use_amp):
                for j in range(config.iters):
                    optimizer.zero_grad()
                    idx = 0
                    while idx < v:
                        min_i = idx
                        max_i = min(min_i + config.batch_size, v)
                        vertices_subset = vertices[min_i:max_i].detach()
                        vertices_subset.requires_grad_()
                        pred_sdf = model(vertices_subset)
                        normals_subset = gradient(pred_sdf, vertices_subset).detach()
                        eikonal = eikonal_loss(normals_subset[..., :3]) / n_batches
                        # mape_ls = mape_loss(pred_sdf, gt_sdf[min_i:max_i]) / n_batches
                        # d_Phi / d_t
                        loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches
                        avg_dict["sdf_l1_loss"] += loss.item()
                        # Time smoothing regularization
                        # if e % config.time_smoothing_period == 0:
                        #     time_grad_loss = time_smoothing_inter_frames_loss(
                        #         t, t + 1, model, config, mesh_res
                        #     )
                        #     loss = loss + config.time_smoothing_lam * time_grad_loss
                        loss = loss + config.eikonal_lam * eikonal
                        # loss = loss + config.mape_lam * mape_ls
                        accelerator.backward(loss)
                        # avg_dict["mape_loss"] += mape_ls.item()
                        avg_dict["sdf_loss"] += loss.item()
                        avg_dict["eikonal_loss"] += eikonal.item()
                        idx += config.batch_size
                    plot_grad_flow(model.named_parameters(), logger, e)
                    optimizer.step()

            if e % config.mesh_log_freq == 0:
                mesh = trimesh.Trimesh(vertices_np, faces_np)
                cd = compute_trimesh_chamfer(R.mesh, mesh)
                avg_dict["cd_loss"] += cd

        if e < config.warmup_epochs:
            lr_scheduler["warmup_scheduler"].step()
        if e >= config.warmup_epochs:
            lr_scheduler["scheduler"].step(e - config.warmup_epochs)

        logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=e)
        logger.add_scalar(
            "image loss",
            avg_dict["photometric_loss"] / config.num_frames,
            global_step=e,
        )
        logger.add_scalar(
            "image loss/mse",
            avg_dict["image_mse_loss"] / config.num_frames,
            global_step=e,
        )
        if config.ms_ssim_lam > 0:
            logger.add_scalar(
                "image loss/ms_ssim",
                avg_dict["mssim_loss"] / config.num_frames,
                global_step=e,
            )
        logger.add_scalar(
            "image loss/laplace",
            avg_dict["laplace_loss"] / config.num_frames,
            global_step=e,
        )

        logger.add_scalar(
            "sdf_loss", avg_dict["sdf_loss"] / config.num_frames, global_step=e
        )

        logger.add_scalar(
            "sdf_loss/L1", avg_dict["sdf_l1_loss"] / config.num_frames, global_step=e
        )

        logger.add_scalar(
            "sdf_loss/eikonal",
            avg_dict["eikonal_loss"] / config.num_frames,
            global_step=e,
        )

        # logger.add_scalar(
        #     "sdf_loss/mape",
        #     avg_dict["mape_loss"] / config.num_frames,
        #     global_step=e,
        # )

        if e % config.video_log_freq == 0:
            images_ = torch.stack(images)
            video_tensor = images_.permute(0, 3, 1, 2).unsqueeze(0)
            logger.add_video(
                "video",
                video_tensor,
                global_step=(e),
                fps=10,
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
                "est/RGB",
                est_grid,
                global_step=(e),
            )
            if config.include_depth:
                logger.add_image(
                    "est/Depth",
                    depth_grid,
                    global_step=(e),
                )

        if e % config.mesh_log_freq == 0:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(
                    torch.tensor(avg_dict["psnr"] / config.num_frames)
                )
                logger.add_scalar("psnr", psnr, global_step=(e))
                logger.add_scalar(
                    "cd", avg_dict["cd_loss"] / config.num_frames, global_step=(e)
                )

        if e % config.mesh_save_freq == 0:
            mesh = trimesh.Trimesh(vertices_np, faces_np)
            mesh.export(f"{config.expdir}/mesh_{e}.ply")

        if e % config.ckpt_log_freq == 0:
            os.makedirs(f"{config.expdir}/iter_{(e):07d}", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{config.expdir}/iter_{(e):07d}/model.pth",
            )
            torch.save(
                optimizer.state_dict(),
                f"{config.expdir}/iter_{(e):07d}/optim.pth",
            )
            torch.save(
                lr_scheduler["warmup_scheduler"].state_dict(),
                f"{config.expdir}/iter_{(e):07d}/warmup_sched.pth",
            )
            torch.save(
                lr_scheduler["scheduler"].state_dict(),
                f"{config.expdir}/iter_{(e):07d}/lr_sched.pth",
            )

        if e % config.evaluate_intermediate_times_every == 0:
            for t in np.arange(0, config.num_frames - 1, 0.5):
                if t % 1 == 0:  # simply, if t is an integer
                    continue
                try:
                    with torch.no_grad():
                        vertices_np, faces_np = model.get_zero_points(
                            mesh_res=mesh_res,
                            t=t,
                            device=device,
                            batch_size=config.batch_size_for_mc,
                        )
                except:
                    print(f"MC Failed for t={t} in epoch {e}. Skipping...")
                    continue

                with torch.no_grad():
                    v_temp = torch.from_numpy(vertices_np.copy()).to(device).float()
                    f_temp = torch.from_numpy(faces_np.copy()).to(device).int()
                    face_normals_temp = compute_face_normals(v_temp, f_temp)
                    vertex_normals_temp = compute_vertex_normals(
                        v_temp, f_temp, face_normals_temp
                    )
                    imgs_temp = R.render(
                        v_temp, f_temp, vertex_normals_temp, device=device
                    )
                    img = imgs_temp[-1][..., :3]
                    logger.add_image(
                        f"intermediate_times/{t}",
                        img.permute(2, 0, 1).clamp(0, 1),
                        global_step=e,
                    )
