import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import make_grid
from render import Renderer, DatasetRenderer
from utils import *
from loss import *
import matplotlib.pyplot as plt
from render_utils import *
from data_utils import *
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
import nvdiffrast.torch as dr
import imageio
from torchvision import transforms
from rendering_gaussian import render, Camera, render_with_seperate_args


def perform_training(
    config,
    device,
    model,
    render_model,
    logger,
    ssim,
    optimizer,
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
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)
    damping_too_small = False
    gif_frames = []
    render_optimizer = torch.optim.Adam(
        render_model.parameters(),
        lr=config.render_lr,
        betas=(0.9, 0.99),
        eps=1e-15,
    )

    renderers = []
    cameras = {}
    global_context = dr.RasterizeGLContext()
    if config.dataset_type == "moving_person":
        dataset = MovingPersonDataset("data/MovingPerson")
        for i in range(len(dataset)):
            imgs, w2cs, intrinsics, from_na = dataset[i]
            with torch.no_grad():
                temp_r = DatasetRenderer(imgs, w2cs, intrinsics, from_na=from_na)
            renderers.append(temp_r)
    elif config.dataset_type == "synthetic":
        for i in range(config.num_frames):
            name = (
                f"{i + 1}.obj"
                if os.path.exists(config.mesh + f"{i + 1}.obj")
                else (
                    f"{i + 1}.ply"
                    if os.path.exists(config.mesh + f"{i + 1}.ply")
                    else ""
                )
            )
            if name == "":
                print(
                    f"Mesh file for frame {i + 1} not found in {config.mesh}. Assuming the same mesh (t=1) for all frames."
                )
                name = "1.obj" if os.path.exists(config.mesh + "1.obj") else "1.ply"

            with torch.no_grad():
                temp_r = Renderer(
                    config.num_views,
                    config.res,
                    glctx=global_context,
                    fname=config.mesh + name,
                    scale=config.scale,
                )

            renderers.append(temp_r)

    for i, renderer in enumerate(renderers):
        if not i in cameras.keys():
            cameras[i] = []

        for j in range(config.num_views):
            if j >= len(cameras[i]):
                cameras[i].append(
                    Camera(
                        renderer.fov_x,
                        renderer.fov_x,
                        config.res,
                        config.res,
                        renderer.view_mats[j],
                        renderer.mvps[j],
                        renderer.r_campos[j],
                    )
                )

    # Logging the target images for all frames
    for i, R in enumerate(renderers):
        target_imgs = R.target_imgs
        for j in range(target_imgs.shape[0]):
            logger.add_image(
                f"targets/target_{i}",
                target_imgs[j][..., :3].permute(2, 0, 1).clamp(0, 1),
                global_step=j,
            )
            if target_imgs.shape[-1] == 4:
                target_depths = target_imgs[j][..., 3]
                logger.add_image(
                    f"targets/target_depth_{i}",
                    target_depths.unsqueeze(0),
                    global_step=j,
                )

    for e in qbar:
        qbar.set_description(f"Epoch {e}")
        # prof_.step()

        laplace_lam = config.max_laplace_lam

        if e >= config.fine_e:
            laplace_lam = config.min_laplace_lam
            mesh_res = config.mesh_res_limit + np.random.randint(low=-3, high=3)

        else:
            laplace_lam = config.max_laplace_lam
            mesh_res = config.mesh_res_base + np.random.randint(low=-3, high=3)

        images = []
        # depths = []
        avg_metrics_dict = {
            "photometric_loss": 0,
            "colored_loss": 0,
            "sdf_l1_loss": 0,
            "sdf_loss": 0,
            "df_dt_loss": 0,
            "cd_loss": 0,
            "laplace_loss": 0,
            "ssim_loss": 0,
            "eikonal_loss": 0,
            "image_mse_loss": 0,
            "psnr": 0,
            "ssim": 0,
            "lpips": 0,
            "splats_sdf_loss": 0,
        }

        # Iterating over frames (in one epoch)
        for t in range(config.num_frames):
            qbar.set_postfix(frame=t)
            R = renderers[t]
            cameras_lst = cameras[t]
            target_imgs = R.target_imgs

            # Normalizing t to [0, 1]
            t = t / (config.num_frames - 1) if config.num_frames > 1 else t

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

            vertices.grad = None

            edges = compute_edges(vertices[:v, :3], faces[:f])
            L = laplacian_simple(vertices[:v, :3], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v, :3]).T @ vertices[:v, :3]))

            face_normals = compute_face_normals(vertices[:v, :3], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v, :3], faces[:f], face_normals
            )

            imgs = R.render(vertices[:v, :3], faces[:f], vertex_normals, device=device)
            plt.imsave(
                f"{config.expdir}/est_geom.png",
                imgs[-1][..., :3].detach().cpu().numpy().clip(0, 1),
                cmap="gray",
            )
            plt.imsave(
                f"{config.expdir}/gt_geom.png",
                target_imgs[-1][..., :3].detach().cpu().numpy().clip(0, 1),
                cmap="gray",
            )
            # plt.imsave(
            #     f"{config.expdir}/est_colored.png",
            #     colored_imgs[-1].detach().cpu().numpy().clip(0, 1),
            # )

            # images.append(imgs[-1][..., 3].detach().cpu())
            # depths.append(imgs[-1][..., 3].detach().cpu())

            # Computing E
            loss = img_loss(
                imgs,
                target_imgs,
                multi_scale=True,
                include_depth=config.include_depth,
                config=config,
                device=device,
                foreground_only=False,
            )

            avg_metrics_dict["photometric_loss"] += loss.item()
            avg_metrics_dict["psnr"] += loss.item()
            loss = loss + laplace_lam * laplacian_loss
            loss.backward()
            avg_metrics_dict["laplace_loss"] += laplacian_loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                os.makedirs(f"{config.expdir}/best", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"{config.expdir}/best/model.pth",
                )
                torch.save(
                    optimizer.state_dict(),
                    f"{config.expdir}/best/optim.pth",
                )
                if config.with_texture:
                    torch.save(
                        render_model.state_dict(),
                        f"{config.expdir}/best/render_model.pth",
                    )

            with torch.no_grad():
                dE_dx = vertices.grad[
                    :v
                ].detach()  # NOTE: dE / dx = - dx / dt (refer to the paper)

            idx = 0
            while idx < v:
                optimizer.zero_grad()
                min_i = idx
                max_i = min(min_i + config.batch_size, v)
                vertices_subset = vertices[min_i:max_i]
                vertices_subset.requires_grad_()
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=use_amp
                ):
                    pred_sdf, _ = model(vertices_subset)
                normals[min_i:max_i] = gradient(pred_sdf, vertices_subset).detach()

                # Flow field (F) = sum of (d phi/dx * - dx / dt)
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
                    vertices_subset = vertices[min_i:max_i]
                    with torch.autocast(
                        device_type=device, dtype=torch.bfloat16, enabled=use_amp
                    ):
                        pred_sdf, _ = model(vertices_subset.detach())
                        # d_Phi / d_t
                        loss = (gt_sdf[min_i:max_i] - pred_sdf).abs().mean() / n_batches
                        avg_metrics_dict["sdf_l1_loss"] += loss.item()

                        pred_sdf, _ = model(vertices_subset)
                        grads = gradient(pred_sdf, vertices_subset)
                        if config.initial_df_dt_lam > 0 and not damping_too_small:
                            damp_factor = get_damping_factor(
                                e, config.initial_df_dt_lam, 0.995, "exponential"
                            )
                            loss += (
                                damp_factor * (grads[:, 3].abs().mean()) / n_batches
                            )  # df/dt
                            avg_metrics_dict["df_dt_loss"] += loss.item()
                            if damp_factor < 5e-8:
                                damping_too_small = True

                        if config.eikonal_lam > 0:
                            eikonal = (
                                grads[..., :3].norm(dim=-1) - 1
                            ).square().mean() / n_batches
                            loss = loss + config.eikonal_lam * eikonal

                    grad_scaler.scale(loss).backward()
                    avg_metrics_dict["sdf_loss"] += loss.item()
                    avg_metrics_dict["eikonal_loss"] += (
                        eikonal.item() if config.eikonal_lam > 0 else 0
                    )
                    idx += config.batch_size

                grad_scaler.step(optimizer)
                grad_scaler.update()

            if e % config.mesh_log_freq == 0:
                if config.dataset_type == "synthetic":
                    mesh = trimesh.Trimesh(vertices_np, faces_np)
                    cd = compute_trimesh_chamfer(R.mesh, mesh)
                    avg_metrics_dict["cd_loss"] += cd
                else:
                    avg_metrics_dict["cd_loss"] += 0

            if e % config.mesh_save_freq == 0 or e == config.epochs - 1:
                mesh = trimesh.Trimesh(vertices_np, faces_np)
                mesh.export(f"{config.expdir}/mesh_{e}_{t}.ply")

        logger.add_scalar(
            "image loss",
            avg_metrics_dict["photometric_loss"] / config.num_frames,
            global_step=e,
        )
        logger.add_scalar(
            "image loss/mse",
            avg_metrics_dict["image_mse_loss"] / config.num_frames,
            global_step=e,
        )
        if config.ssim_lam > 0:
            logger.add_scalar(
                "image loss/ssim",
                avg_metrics_dict["ssim_loss"] / config.num_frames,
                global_step=e,
            )
        logger.add_scalar(
            "image loss/laplace",
            avg_metrics_dict["laplace_loss"] / config.num_frames,
            global_step=e,
        )

        logger.add_scalar(
            "sdf_loss", avg_metrics_dict["sdf_loss"] / config.num_frames, global_step=e
        )

        logger.add_scalar(
            "sdf_loss/L1",
            avg_metrics_dict["sdf_l1_loss"] / config.num_frames,
            global_step=e,
        )

        logger.add_scalar(
            "sdf_loss/eikonal",
            avg_metrics_dict["eikonal_loss"] / config.num_frames,
            global_step=e,
        )

        logger.add_scalar(
            "sdf_loss/df_dt",
            avg_metrics_dict["df_dt_loss"] / config.num_frames,
            global_step=e,
        )

        logger.add_scalar(
            "sdf_loss/splats_sdf_loss",
            avg_metrics_dict["splats_sdf_loss"] / config.num_frames,
            global_step=e,
        )

        # if e % config.img_log_freq == 0:
        # est_grid = make_grid(images[0][..., :3].clamp(0, 1))
        # # depth_grid = make_grid(depths[0].unsqueeze(0).clamp(0, 1))
        # for img in images[1:]:
        #     est_grid = torch.cat(
        #         (est_grid, make_grid(img[..., :3].clamp(0, 1))),
        #         dim=1,
        #     )
        # # for depth in depths[1:]:
        # #     depth_grid = torch.cat(
        # #         (depth_grid, make_grid(depth.unsqueeze(0).clamp(0, 1))),
        # #         dim=2,
        # #     )
        # est_grid = est_grid.permute(2, 0, 1)  # reshape into (C, H, W)
        # logger.add_image(
        #     "est/RGB",
        #     est_grid,
        #     global_step=(e),
        # )
        # logger.add_image(
        #     "est/colored",
        #     colored_imgs[-1].permute(2, 0, 1).clamp(0, 1),
        #     global_step=(e),
        # )
        # if config.include_depth:
        #     logger.add_image(
        #         "est/Depth",
        #         depth_grid,
        #         global_step=(e),
        #     )

        # est_grid = transforms.ToPILImage()(est_grid)
        # gif_frames.append(est_grid)

        # imageio.mimsave(f"{config.expdir}/gif.gif", gif_frames, fps=5)

        if e % config.mesh_log_freq == 0:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(
                    torch.tensor(avg_metrics_dict["psnr"] / config.num_frames)
                )
                logger.add_scalar("psnr", psnr, global_step=(e))
                logger.add_scalar(
                    "cd",
                    avg_metrics_dict["cd_loss"] / config.num_frames,
                    global_step=(e),
                )
                logger.add_scalar(
                    "ssim",
                    avg_metrics_dict["ssim"] / config.num_frames,
                    global_step=(e),
                )

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
            if config.with_texture:
                torch.save(
                    render_model.state_dict(),
                    f"{config.expdir}/iter_{(e):07d}/render_model.pth",
                )

        # if e % config.evaluate_intermediate_times_every == 0:
        #     for t in np.arange(0, config.num_frames - 1, 0.5):
        #         if t % 1 == 0:  # simply, if t is an integer
        #             continue
        #         try:
        #             t = t / (config.num_frames - 1)
        #             with torch.no_grad():
        #                 vertices_np, faces_np = model.get_zero_points(
        #                     mesh_res=mesh_res,
        #                     t=t,
        #                     device=device,
        #                     batch_size=config.batch_size_for_mc,
        #                 )
        #         except:
        #             print(f"MC Failed for t={t} in epoch {e}. Skipping...")
        #             continue

        #         with torch.no_grad():
        #             v_temp = torch.from_numpy(vertices_np.copy()).to(device).float()
        #             f_temp = torch.from_numpy(faces_np.copy()).to(device).int()
        #             face_normals_temp = compute_face_normals(v_temp, f_temp)
        #             vertex_normals_temp = compute_vertex_normals(
        #                 v_temp, f_temp, face_normals_temp
        #             )
        #             imgs_temp = R.render(
        #                 v_temp, f_temp, vertex_normals_temp, device=device
        #             )
        #             img = imgs_temp[-1][..., :3]
        #             logger.add_image(
        #                 f"intermediate_times/{t}",
        #                 img.permute(2, 0, 1).clamp(0, 1),
        #                 global_step=e,
        #             )

        del images  # , depths
    if gif_frames:
        imageio.mimsave(f"{config.expdir}/gif.gif", gif_frames, fps=5)
        print(f"GIF saved to {config.expdir}/gif.gif")

    print(f"Finished training for {e} epochs")
    return best_loss
