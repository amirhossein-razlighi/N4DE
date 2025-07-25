import torch
import numpy as np
from rendering_gaussian import Camera, render_with_seperate_args
from render import Renderer, DatasetRenderer
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from utils import *
from loss import *
from data_utils import MovingPersonDataset


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
    sdf_head = model
    damping_too_small = False

    renderers = []
    cameras = {}
    global_context = dr.RasterizeCudaContext()  # dr.RasterizeGLContext()
    if config.dataset_type == "moving_person":
        dataset = MovingPersonDataset("data/MovingPerson")
        for i in range(min(config.num_frames, len(dataset))):
            imgs, c2ws, intrinsics, from_na, scale, offset = dataset[i]
            temp_r = DatasetRenderer(
                imgs,
                c2ws,
                intrinsics,
                from_na=from_na,
                scale=scale,
                offset=offset,
                config=config,
            )
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

            temp_r = Renderer(
                config.num_views,
                config.res,
                glctx=global_context,
                fname=config.mesh + name,
                scale=config.scale,
                with_texture=config.with_texture,
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
                        renderer.fov_y,
                        config.res,
                        config.res,
                        renderer.view_mats[j],
                        renderer.mvps[j],
                        renderer.view_mats[j][:3, 3],  # 1
                        # renderer.camera_positions[j],  # 2
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

    render_head = render_model
    render_optimizer = torch.optim.Adam(render_head.parameters(), lr=config.render_lr)
    sdf_optimizer = optimizer

    if config._continue and config.with_texture:
        render_optimizer.load_state_dict(
            torch.load(
                f"{config.expdir}/checkpoints/render_optimizer_{config.last_epoch}.pth"
            )
        )

    best_loss_splat = torch.inf
    best_loss_sdf = torch.inf

    for e in qbar:
        for t in range(config.num_frames):
            R = renderers[t]
            cameras_lst = cameras[t]
            targets = R.target_imgs

            # Normalizing t to [0, 1]
            t = t / (config.num_frames - 1) if config.num_frames > 1 else t

            with torch.no_grad():
                vertices_np, faces_np = sdf_head.get_zero_points(
                    mesh_res=config.mesh_res_base,
                    t=t,
                    device="cuda",
                    batch_size=config.batch_size_for_mc,
                )

                v = vertices_np.shape[0]
                f = faces_np.shape[0]
                concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
                vertices.data[:v] = torch.from_numpy(concated).float()
                faces.data[:f] = torch.from_numpy(np.ascontiguousarray(faces_np))

            vertices.grad = None

            face_normals = compute_face_normals(vertices[:v, :3], faces[:f])
            vertex_normals = compute_vertex_normals(
                vertices[:v, :3], faces[:f], face_normals
            )
            edges = compute_edges(vertices[:v, :3], faces[:f])
            L = laplacian_simple(vertices[:v, :3], edges.long())
            laplacian_loss = torch.trace(((L @ vertices[:v, :3]).T @ vertices[:v, :3]))

            if config.with_texture:
                sampled_cams = np.random.choice(
                    len(cameras_lst), config.num_sample_views_for_splats
                )
                render_loss = 0
                bg = torch.tensor([0.0, 0.0, 0.0]).float().cuda()

                for j in sampled_cams:
                    render_optimizer.zero_grad()
                    shs_list = []
                    opacs_list = []
                    scals_list = []
                    rots_list = []

                    cam = cameras_lst[j]

                    idx = 0
                    batch_size = 100_000

                    while idx < v:
                        min_idx = idx
                        max_idx = min(idx + batch_size, v)
                        vertices_subset = vertices[min_idx:max_idx].detach()

                        sh_coeffs, opacity, scaling, rotation = render_head(
                            vertices_subset
                        )

                        shs_list.append(sh_coeffs)
                        opacs_list.append(opacity)
                        scals_list.append(scaling)
                        rots_list.append(rotation)

                        idx += batch_size

                    shs = torch.cat(shs_list, dim=0).cuda()
                    opacs = torch.cat(opacs_list, dim=0).cuda()
                    scals = torch.cat(scals_list, dim=0).cuda()
                    rots = torch.cat(rots_list, dim=0).cuda()

                    shs_list.clear()
                    opacs_list.clear()
                    scals_list.clear()
                    rots_list.clear()

                    try:
                        img = render_with_seperate_args(
                            cam,
                            vertices[:v, :3].detach(),
                            3,
                            opacs,
                            scals,
                            rots,
                            shs,
                            bg,
                            scaling_modifier=config.scale,
                        )["render"].permute(1, 2, 0)
                    except:
                        print("Error in rendering")

                    loss = (
                        nn.functional.l1_loss(img, targets[j][..., :3])
                        + 0.01 * ssim(img, targets[j][..., :3])
                        + 0.05 * nn.functional.mse_loss(img, bg)
                    )

                    render_loss += loss / config.num_sample_views_for_splats

                    if e % 50 == 0:
                        if render_loss < best_loss_splat:
                            best_loss_splat = render_loss
                            torch.save(
                                render_head.state_dict(),
                                f"{config.expdir}/render_head.pth",
                            )

                        logger.add_image(
                            "Render Head/splats",
                            img.detach().cpu().numpy().clip(0, 1),
                            global_step=e,
                            dataformats="HWC",
                        )
                        logger.add_scalar(
                            "Render Head/loss",
                            render_loss.item(),
                            global_step=e,
                        )

                    loss.backward()
                    render_optimizer.step()

                qbar.set_postfix(
                    {
                        "Epoch": e,
                        "Render Head Loss": render_loss.item(),
                    }
                )

            imgs = R.render(vertices[:v, :3], faces[:f], vertex_normals)

            if config.with_texture:
                loss = config.ssim_lam * ssim(
                    imgs[..., :3], targets[..., :3]
                ) + img_loss(imgs, targets, foreground_only=False)
            else:
                loss = img_loss(
                    imgs,
                    targets,
                    multi_scale=True,
                    include_depth=config.include_depth,
                    config=config,
                    device="cuda",
                    foreground_only=False,
                )

            if e > config.fine_e:
                loss = loss + config.min_laplace_lam * laplacian_loss
            else:
                loss = loss + config.max_laplace_lam * laplacian_loss

            loss.backward()
            logger.add_scalar("SDF/Photometric Loss", loss.item(), global_step=e)

            if e % 50 == 0:
                if loss < best_loss_sdf:
                    best_loss_sdf = loss
                    torch.save(sdf_head.state_dict(), f"{config.expdir}/model.pth")

            with torch.no_grad():
                dE_dx = vertices.grad[:v].detach()

            idx = 0
            batch_size = config.batch_size

            while idx < v:
                min_idx = idx
                max_idx = min(idx + batch_size, v)
                vertices_subset = vertices[min_idx:max_idx]
                vertices_subset.requires_grad_()

                pred_sdf, _ = sdf_head(vertices_subset)
                normals[min_idx:max_idx] = gradient(pred_sdf, vertices_subset).detach()

                F[min_idx:max_idx] = torch.nan_to_num(
                    torch.sum(
                        normals[min_idx:max_idx] * dE_dx[min_idx:max_idx],
                        dim=-1,
                        keepdim=True,
                    )
                )

                gt_sdf[min_idx:max_idx] = (
                    pred_sdf + config.eps * F[min_idx:max_idx]
                ).detach()

                idx += batch_size

            n_batches = v // config.batch_size
            if n_batches == 0:
                n_batches = 1

            sdf_optimizer.zero_grad()
            idx = 0
            eikonal_acc = 0
            sdf_loss_acc = 0

            while idx < v:
                min_idx = idx
                max_idx = min(idx + batch_size, v)
                vertices_subset = vertices[min_idx:max_idx]

                pred_sdf, _ = sdf_head(vertices_subset)

                loss = (gt_sdf[min_idx:max_idx] - pred_sdf).abs().mean() / n_batches
                grads = gradient(pred_sdf, vertices_subset)
                eikonal = (grads[..., :3].norm(dim=-1) - 1).square().mean() / n_batches
                eikonal_acc += eikonal.item()
                grads = gradient(pred_sdf, vertices_subset)
                
                if config.initial_df_dt_lam > 0 and not damping_too_small:
                    damp_factor = get_damping_factor(
                        e, config.initial_df_dt_lam, 0.995, "exponential"
                    )
                    loss += (
                        damp_factor * (grads[:, 3].abs().mean()) / n_batches
                    )  # df/dt
                    if damp_factor < 5e-8:
                         damping_too_small = True

                loss = loss + config.eikonal_lam * eikonal
                sdf_loss_acc += loss.item()
                loss.backward()
                idx += batch_size

            sdf_optimizer.step()

            if e % 50 == 0:
                qbar.set_postfix(
                    {
                        "Epoch": e,
                        "SDF Total Loss": sdf_loss_acc,
                        "Eikonal Loss": eikonal_acc,
                        "MSE Loss": torch.nn.functional.mse_loss(
                            imgs[..., :3], targets[..., :3]
                        ).item(),
                        "PSNR": -10
                        * torch.log10(
                            torch.nn.functional.mse_loss(
                                imgs[..., :3], targets[..., :3]
                            )
                        ).item(),
                    }
                )
                logger.add_image(
                    "SDF",
                    imgs[-1][..., :3].detach().cpu().numpy().clip(0, 1),
                    global_step=e,
                    dataformats="HWC",
                )
                logger.add_scalar(
                    "SDF/Eikonal Loss",
                    eikonal_acc,
                    global_step=e,
                )
                logger.add_scalar(
                    "SDF/MSE Loss",
                    torch.nn.functional.mse_loss(
                        imgs[..., :3], targets[..., :3]
                    ).item(),
                    global_step=e,
                )
                logger.add_scalar(
                    "SDF/PSNR",
                    -10
                    * torch.log10(
                        torch.nn.functional.mse_loss(imgs[..., :3], targets[..., :3])
                    ).item(),
                    global_step=e,
                )
                logger.add_scalar(
                    "SDF/Total Loss",
                    sdf_loss_acc,
                    global_step=e,
                )

                if config.with_texture:
                    with torch.no_grad():
                        sh_coeffs__ = []
                        idx = 0

                        while idx < v:
                            end_idx = min(idx + config.batch_size, v)
                            sh, _, _, _ = render_head(vertices[idx:end_idx].detach())
                            sh_coeffs__.append(sh)
                            idx += config.batch_size

                        sh_coeffs__ = torch.cat(sh_coeffs__, dim=0).cuda()

                        shs_view = sh_coeffs__.transpose(1, 2).view(-1, 3, (3 + 1) ** 2)

                        mean_loss_photometric = 0

                        for view_num in range(config.num_views):
                            dir_pp = vertices[:v, :3] - R.camera_positions[
                                view_num
                            ].repeat(sh_coeffs__.shape[0], 1)
                            dir_pp_normalized = dir_pp / dir_pp.norm(
                                dim=1, keepdim=True
                            )
                            sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
                            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                            est = R.render_coloured(
                                vertices[:v, :3],
                                faces[:f],
                                vertex_normals,
                                vertex_colors=colors_precomp,
                                view_idx=view_num,
                                # albedo=1.0,
                            )[0][..., :3]

                            mean_loss_photometric += (
                                nn.functional.mse_loss(est, targets[view_num][..., :3])
                                / config.num_views
                            )

                            colored_final = est.cpu().detach().numpy().clip(0, 1)

                            logger.add_image(
                                "Final Colored/est",
                                colored_final,
                                global_step=view_num,
                                dataformats="HWC",
                            )

                        logger.add_scalar(
                            "Final Colored/Photometric loss",
                            mean_loss_photometric.item(),
                            global_step=e,
                        )
                        logger.add_scalar(
                            "Final Colored/PSNR",
                            -10 * torch.log10(mean_loss_photometric).item(),
                            global_step=e,
                        )

            if e % 100 == 0:
                torch.save(
                    sdf_head.state_dict(),
                    f"{config.expdir}/checkpoints/model_{e}.pth",
                )
                torch.save(
                    render_head.state_dict(),
                    f"{config.expdir}/checkpoints/render_head_{e}.pth",
                )
                torch.save(
                    render_optimizer.state_dict(),
                    f"{config.expdir}/checkpoints/render_optimizer_{e}.pth",
                )
                torch.save(
                    sdf_optimizer.state_dict(),
                    f"{config.expdir}/checkpoints/sdf_optimizer_{e}.pth",
                )
