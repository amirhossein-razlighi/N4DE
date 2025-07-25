from models_gaussian import *
from rendering_gaussian import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models_animated import EncoderPlusSDF
from render import Renderer
import warnings
from math import exp

warnings.filterwarnings("ignore")


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


def train():
    encoder_with_sdf_model = EncoderPlusSDF().cuda()
    encoder_with_sdf_model.load_state_dict(
        torch.load("logs/127_Encoder/best/model.pth")
    )
    encoder_with_sdf_model.eval()

    vertices_np, _ = encoder_with_sdf_model.get_zero_points(t=0, mesh_res=150)
    vertices = torch.from_numpy(vertices_np).float().cuda()
    sample_red_colors = (
        torch.ones((vertices.shape[0], 3)).cuda()
        * torch.tensor([1, 0, 0]).float().cuda()
    )
    pcd = BasicPointCloud(vertices, sample_red_colors)

    gaussian_model = GaussianModel(3)  # SH degree = 3
    options = OptimizationParams()
    gaussian_model.create_from_pcd(pcd, options.position_lr_init)
    gaussian_model.training_setup(options)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    renderer = Renderer(100, 128, "data/Textured_Bunny/bunny.obj", with_texture=True)
    gt_imgs = renderer.target_imgs[..., :3]
    cameras = [
        Camera(renderer.fov_x, renderer.fov_x, 128, 128, mvs, mvp, campos)
        for mvs, mvp, campos in zip(
            renderer.view_mats, renderer.mvps, renderer.r_campos
        )
    ]
    losses = []
    for iteration in range(0, options.iterations + 1):
        gaussian_model.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()

        rand_idx = np.random.randint(0, len(cameras))
        random_camera = cameras[rand_idx]
        target_img = gt_imgs[rand_idx]
        # bg = torch.rand((3), device="cuda")  # Random background color
        render_pkg = render(random_camera, gaussian_model, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        image = image.permute(1, 2, 0)
        plt.imsave(f"GS_Results/est.png", image.cpu().detach().numpy().clip(0, 1))
        plt.imsave(f"GS_Results/gt.png", target_img.cpu().detach().numpy().clip(0, 1))
        l1_loss = nn.L1Loss(reduction="mean")(image, target_img)
        loss = (1.0 - options.lambda_dssim) * l1_loss + options.lambda_dssim * (
            1.0 - ssim(image.permute(2, 0, 1), target_img.permute(2, 0, 1))
        )
        loss.backward()
        losses.append(l1_loss.item())

        with torch.no_grad():
            if iteration < options.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussian_model.max_radii2D[visibility_filter] = torch.max(
                    gaussian_model.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussian_model.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > options.densify_from_iter
                    and iteration % options.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > options.opacity_reset_interval else None
                    )
                    gaussian_model.densify_and_prune(
                        options.densify_grad_threshold,
                        0.005,
                        2,  # DEBUG: IDK! 2 because of [-1, 1] scene?
                        size_threshold,
                    )

                if iteration % options.opacity_reset_interval == 0:
                    gaussian_model.reset_opacity()

                if iteration < options.iterations:
                    gaussian_model.optimizer.step()
                    gaussian_model.optimizer.zero_grad(set_to_none=True)

                if iteration % 50 == 0:
                    print(f"Iteration {iteration}, L1 loss avg: {np.mean(losses)}")
                    plt.plot(losses)
                    plt.savefig("GS_Results/loss.png")

        plt.close("all")


if __name__ == "__main__":
    train()
