import torch
from torch.functional import F
from utils import *


def eikonal_loss(spatial_grads):
    gradient_norm = torch.norm(spatial_grads, dim=-1)
    return ((gradient_norm - 1) ** 2).mean()


def orthogonal_normals(normals, Flow_field):
    return torch.sum(normals[..., :3] * Flow_field, dim=-1).abs().mean()


def time_smoothing_inter_frames_loss(current_t, next_t, model, vertices):
    # df/dt = 0 for time between t=i and t=i+1
    frame_to_inspect = (current_t + next_t) / 2
    new_vertices = vertices.clone()
    new_vertices[:, 3] = frame_to_inspect
    pred_sdf = model(new_vertices.unsqueeze(0)).squeeze(0)
    time_grad_forward = torch.autograd.grad(
        pred_sdf,
        new_vertices,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        allow_unused=True,
    )[0][:, 3].detach()
    return time_grad_forward.abs().mean()


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


def img_loss(
    imgs,
    target_imgs,
    multi_scale=True,
    include_depth=False,
    config=None,
    device=torch.device("cuda:0"),
):
    loss = 0
    kernel = gauss_kernel(device=device)
    count = 0
    images = imgs
    imgs = images[:, :, :, :3]
    depths = images[:, :, :, 3]
    targets = target_imgs
    target_imgs = targets[:, :, :, :3]
    target_depths = targets[:, :, :, 3]

    for i in range(imgs.shape[0]):
        count += 1
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
        if include_depth:
            loss += (depths[i] - target_depths[i]).square().mean() * config.lambda_depth
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
