import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from render_utils import eval_sh, SH2RGB, RGB2SH
from render import Renderer
from models_animated import EncoderPlusSDF
from utils import *
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_from_checkpoint(checkpoint_address, model):
    checkpoint = torch.load(checkpoint_address)
    model.load_state_dict(checkpoint)
    return model


class SHPredictor(nn.Module):
    def __init__(self, input_dim, sh_order):
        """
        Initialize the SHPredictor module.

        Parameters:
        - input_dim: int, the dimension of the input features.
        - sh_order: int, the order of spherical harmonics.
        """
        super(SHPredictor, self).__init__()
        self.sh_order = sh_order
        self.num_sh_coeffs = (sh_order + 1) ** 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * self.num_sh_coeffs),  # 3 is for RGB
        )

    def forward(self, x):
        """
        Forward pass of the SHPredictor.

        Parameters:
        - x: torch.Tensor, the input features, shape (N, input_dim).

        Returns:
        - sh_coeffs: torch.Tensor, the predicted SH coefficients, shape (N, 3, num_sh_coeffs).
        """
        sh_coeffs = self.fc(x)
        sh_coeffs = sh_coeffs.view(-1, 3, self.num_sh_coeffs)
        return sh_coeffs


def render_colors(sh_coeffs, directions, sh_order):
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


def evaluate_model_and_save_images(
    model, sh_model, renderer, num_frames, sh_order, mesh_res
):
    with torch.no_grad():
        for t in range(num_frames):
            vertices_np, faces_np = model.get_zero_points(t, mesh_res)
            vertices = torch.tensor(vertices_np, dtype=torch.float32, device="cuda:0")
            faces = torch.tensor(faces_np.copy(), dtype=torch.int32, device="cuda:0")
            faces_normals = compute_face_normals(vertices, faces)
            vertex_normals = compute_vertex_normals(vertices, faces, faces_normals)
            v = vertices_np.shape[0]
            concated = np.concatenate((vertices_np, np.ones((v, 1)) * 0), axis=1)
            sh_coeffs = sh_model(torch.from_numpy(concated).cuda().float())
            vertex_colors = render_colors(
                sh_coeffs,
                calculate_view_direction(vertices, renderer.camera_positions),
                sh_order,
            )
            imgs = []
            for i in range(renderer.target_imgs.shape[0]):
                imgs.append(
                    renderer.render_coloured(
                        vertices, faces, vertex_normals, i, vertex_colors[i]
                    ).squeeze(0)
                )
            os.makedirs("SH_Results", exist_ok=True)
            plt.imsave(
                f"SH_Results/est_{t}.png", imgs[t].detach().cpu().numpy().clip(0, 1)
            )
            plt.imsave(
                f"SH_Results/gt_{t}.png",
                renderer.target_imgs[t].detach().cpu().numpy().clip(0, 1),
            )


if __name__ == "__main__":
    IMAGE_RES = 100
    SH_DEGREE = 4
    NUM_EPOCHS = 3000
    MESH_RES = 150
    NUM_FRAMES = 2
    NUM_VIEWS = 100

    renderer = Renderer(
        NUM_VIEWS, IMAGE_RES, fname="data/Textured_Bunny/1.obj", with_texture=True
    )
    target_images = renderer.target_imgs
    model = EncoderPlusSDF(4, device="cuda:0")
    model = load_from_checkpoint("logs/5_No_Color/best/model.pth", model).cuda()
    model.eval()

    sh_model = SHPredictor(4, SH_DEGREE).cuda()
    # try:
    #     sh_model = load_from_checkpoint(
    #         "checkpoints/best_sh_model.pth", sh_model
    #     ).cuda()
    #     print("Loaded SH model from checkpoint")
    # except:
    #     pass
    sh_model.train()
    optimizer = torch.optim.Adam(sh_model.parameters(), lr=1e-3)
    best_loss = np.inf
    train_losses = []
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        pbar.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loss = 0
        optimizer.zero_grad()
        for t in range(NUM_FRAMES):
            vertices_np, faces_np = model.get_zero_points(t, MESH_RES)
            vertices = torch.tensor(vertices_np, dtype=torch.float32, device="cuda:0")
            faces = torch.tensor(faces_np.copy(), dtype=torch.int32, device="cuda:0")
            faces_normals = compute_face_normals(vertices, faces)
            vertex_normals = compute_vertex_normals(vertices, faces, faces_normals)
            v = vertices_np.shape[0]
            concated = np.concatenate((vertices_np, np.ones((v, 1)) * t), axis=1)
            sh_coeffs = sh_model(torch.from_numpy(concated).cuda().float())
            vertex_colors = render_colors(
                sh_coeffs,
                calculate_view_direction(vertices, renderer.camera_positions),
                SH_DEGREE,
            )
            imgs = []
            for i in range(target_images.shape[0]):
                imgs.append(
                    renderer.render_coloured(
                        vertices, faces, vertex_normals, i, vertex_colors[i]
                    ).squeeze(0)
                )
            imgs = torch.stack(imgs).cuda()
            loss += torch.nn.functional.mse_loss(imgs, target_images)
            pbar.set_postfix({"t": t, "loss": loss.item(), "best_loss": best_loss})

        if epoch % 25 == 0:
            evaluate_model_and_save_images(
                model, sh_model, renderer, NUM_FRAMES, SH_DEGREE, MESH_RES
            )
        loss /= NUM_FRAMES
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(sh_model.state_dict(), "checkpoints/best_sh_model.pth")

        if epoch % 100 == 0:
            plt.plot(train_losses)
            plt.savefig("SH_Results/train_losses.png")
