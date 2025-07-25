import torch
from torch.functional import F
from utils import *


def hessian_loss(pred_sdf, vertices):
    hessian = gradient(gradient(pred_sdf, vertices), vertices)
    hessian = hessian[:, :3]
    return (hessian.norm(dim=-1) ** 2).mean()


def orthogonal_normals_loss(normals, Flow_field):
    return torch.sum(normals[..., :3] * Flow_field, dim=-1).abs().mean()


def time_smoothing_inter_frames_loss(current_t, next_t, model, config, mesh_res):
    # df/dt = 0 for time between t=i and t=i+1
    frame_to_inspect = (current_t + next_t) / 2
    vertices_np, _ = model.get_zero_points(
        mesh_res=mesh_res,
        t=frame_to_inspect,
        device="cuda",
        batch_size=config.batch_size_for_mc,
    )
    v = vertices_np.shape[0]
    concated = np.concatenate((vertices_np, np.ones((v, 1)) * frame_to_inspect), axis=1)
    verts = torch.from_numpy(concated).to("cuda")

    time_grads = torch.zeros((v, 1)).to("cuda")
    idx = 0
    batch_size = config.batch_size
    while idx < v:
        min_i = idx
        max_i = min(min_i + batch_size, v)
        verts_subset = verts[min_i:max_i]
        verts_subset.requires_grad_()

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=config.use_amp
        ):
            pred_sdf, _ = model(verts_subset.float())
        time_grad = gradient(pred_sdf, verts_subset)[:, 3]  # df/dt
        time_grads[min_i:max_i] = time_grad.unsqueeze(-1)

    return (time_grads**2).mean()


def time_smoothing_loss(time_grad):
    return time_grad.abs().mean()


# term for morphing
# d_phi/dt * <grad(phi), F>
# loss += normals[min_i:max_i, 3].abs().mean() * torch.sum(normals[min_i:max_i, :3] * F[min_i:max_i], dim=-1).abs().mean() / n_batches


def gauss_kernel(size=5, device=torch.device("cuda:0"), channels=3):
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


def conv_gauss(img, kernel, device=torch.device("cuda:0")):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
    out = torch.nn.functional.conv2d(
        img.to(device), kernel.to(device), groups=img.shape[1]
    )
    return out


def mape_loss(pred, target, reduction="mean"):
    # pred, target: [B, 1], torch tensor
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == "mean":
        loss = loss.mean()

    return loss


def foreground_mask(image):
    return (image > 0).any(dim=-1, keepdim=True)


def img_loss(
    imgs,
    target_imgs,
    multi_scale=True,
    include_depth=False,
    config=None,
    device=torch.device("cuda:0"),
    foreground_only=True,
):
    loss = 0
    kernel = gauss_kernel(device=device)
    count = 0
    images = imgs
    imgs = images[..., :3]
    targets = target_imgs
    target_imgs = targets[..., :3]
    try:
        depths = images[..., 3:]
        target_depths = targets[..., 3:]
    except:
        include_depth = False
        # No depth images found, skipping depth loss

    for i in range(imgs.shape[0]):
        count += 1
        if foreground_only:
            fg_mask = foreground_mask(target_imgs[i]) | foreground_mask(imgs[i])
            fg_mask = fg_mask.repeat(1, 1, 3)
            fg_imgs = imgs[i][fg_mask]
            fg_target_imgs = target_imgs[i][fg_mask]
            loss = loss + (fg_imgs - fg_target_imgs).square().mean()
        else:
            loss = loss + (imgs[i] - target_imgs[i]).square().mean()

        if multi_scale:
            current_est = imgs[i].permute(2, 0, 1).unsqueeze(0)
            current_gt = target_imgs[i].permute(2, 0, 1).unsqueeze(0)

            for j in range(4):
                filtered_est = conv_gauss(current_est, kernel, device)
                filtered_gt = conv_gauss(current_gt, kernel, device)
                down_est = downsample(filtered_est)
                down_gt = downsample(filtered_gt)

                current_est = down_est
                current_gt = down_gt

                loss = loss + (current_est - current_gt).square().mean() / (j + 1)
    
    loss = loss / count
    if include_depth:
        loss = (1 - config.lambda_depth) * loss + config.lambda_depth * depth_loss(
            depths, target_depths
        )
    return loss


def depth_loss(
    depth_imgs,
    target_depth_imgs,
    multi_scale=True,
    device=torch.device("cuda:0"),
):
    loss = 0
    kernel = gauss_kernel(device=device, channels=1)  # Use 1-channel kernel for depth
    count = 0

    # Adjust the shape to [batch_size, channels, height, width]
    depth_imgs = depth_imgs.permute(0, 3, 1, 2)
    target_depth_imgs = target_depth_imgs.permute(0, 3, 1, 2)

    for i in range(depth_imgs.shape[0]):
        count += 1
        depth = depth_imgs[i]
        target_depth = target_depth_imgs[i]

        loss += (depth - target_depth).square().mean()

        if multi_scale:
            current_depth = depth.unsqueeze(0)  # Shape to [1, 1, H, W]
            current_target_depth = target_depth.unsqueeze(0)

            for j in range(4):
                filtered_depth = conv_gauss(current_depth, kernel, device)
                filtered_target_depth = conv_gauss(current_target_depth, kernel, device)
                down_depth = downsample(filtered_depth)
                down_target_depth = downsample(filtered_target_depth)

                current_depth = down_depth
                current_target_depth = down_target_depth

                loss += (current_depth - current_target_depth).square().mean() / (j + 1)

    loss = loss / count
    return loss


def loss_morphing(gt_sdf, pred_sdf, F, vertices, t):
    """
    A loss function, doing this:
    for (a,b) -> df/dt + <grad(f), F> = 0
    for t=t_i -> f = g_i
    """
    grad = gradient(pred_sdf, vertices)
    df_dt = grad[:, 3]
    df_3d = grad[:, :3]
    if t == int(t):
        return (gt_sdf - pred_sdf).abs().mean()
    else:
        return (gt_sdf - pred_sdf).abs().mean() + (
            df_dt + torch.sum(df_3d * F, dim=-1)
        ).abs().mean()


def color_diversity_loss(colors):
    return -torch.std(colors, dim=0).mean()
