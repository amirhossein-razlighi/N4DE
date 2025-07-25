import torch
import torch.nn as nn
from grid_encoding import get_a_grid_encoder
from utils import *
from models_animated import EncoderPlusSDF
from render import Renderer
from rendering_gaussian import Camera, render
import nvdiffrast.torch as dr
from simple_knn._C import distCUDA2
from render_utils import *
import matplotlib.pyplot as plt


class OptimizationParams:
    def __init__(self):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False


class RenderHead(nn.Module):
    def __init__(
        self,
        sh_order=3,
        initial_vertices=None,
    ):
        super(RenderHead, self).__init__()
        if initial_vertices is None:
            raise ValueError("initial_vertices must be provided")

        self.sh_order = self.active_sh_degree = sh_order
        self.means3d = nn.Parameter(initial_vertices, requires_grad=True)
        rots = torch.zeros((initial_vertices.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0
        self.rotations = nn.Parameter(rots, requires_grad=True)
        dist2 = torch.clamp_min(
            distCUDA2(initial_vertices),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self.scales = nn.Parameter(scales, requires_grad=True)

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (initial_vertices.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self.opacities = nn.Parameter(opacities, requires_grad=True)

        sample_red_colors = (
            torch.ones((initial_vertices.shape[0], 3)).cuda()
            * torch.tensor([1, 0, 0]).float().cuda()
        )
        fused_color = RGB2SH(sample_red_colors)
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.sh_order + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_xyz(self):
        return self.means3d

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scales)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotations)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacities)


def main():
    test_model = EncoderPlusSDF().cuda()
    test_model.load_state_dict(torch.load("checkpoints/model.pth"))
    test_model.eval()

    vertices_np, _ = test_model.get_zero_points(t=0, mesh_res=150)
    vertices = torch.from_numpy(vertices_np).float().cuda()

    glctx = dr.RasterizeGLContext()
    renderer = Renderer(100, 128, glctx, "data/Textured_Bunny/1.obj", with_texture=True)

    gt_imgs = renderer.target_imgs[..., :3]

    cameras = [
        Camera(renderer.fov_x, renderer.fov_x, 128, 128, mvs, mvp, campos)
        for mvs, mvp, campos in zip(
            renderer.view_mats, renderer.mvps, renderer.r_campos
        )
    ]

    gaussian_model = RenderHead(3, vertices).cuda()
    gaussian_model.train()

    training_args = OptimizationParams()
    spatial_lr_scale = 1.0

    l = [
        {
            "params": [gaussian_model.means3d],
            "lr": training_args.position_lr_init * spatial_lr_scale,
            "name": "xyz",
        },
        {
            "params": [gaussian_model._features_dc],
            "lr": training_args.feature_lr,
            "name": "f_dc",
        },
        {
            "params": [gaussian_model._features_rest],
            "lr": training_args.feature_lr / 20.0,
            "name": "f_rest",
        },
        {
            "params": [gaussian_model.opacities],
            "lr": training_args.opacity_lr,
            "name": "opacity",
        },
        {
            "params": [gaussian_model.scales],
            "lr": training_args.scaling_lr,
            "name": "scaling",
        },
        {
            "params": [gaussian_model.rotations],
            "lr": training_args.rotation_lr,
            "name": "rotation",
        },
    ]

    optimizer = torch.optim.Adam(l, lr=0.0001, eps=1e-15)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    for i in range(100):
        optimizer.zero_grad()

        rand_idx = np.random.randint(0, len(cameras))
        random_camera = cameras[rand_idx]
        target_img = gt_imgs[rand_idx]

        render_pkg = render(random_camera, gaussian_model, background)
        image = render_pkg["render"].permute(1, 2, 0)
        l1_loss = nn.L1Loss(reduction="mean")(image, target_img)
        loss = l1_loss

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Loss: {loss.item()}")
            plt.imsave(
                f"renders/GS/est_{i}.png", image.cpu().detach().numpy().clip(0, 1)
            )
            plt.imsave(
                f"renders/GS/gt_{i}.png", target_img.cpu().detach().numpy().clip(0, 1)
            )


if __name__ == "__main__":
    main()
