import os
import matplotlib.pyplot as plt
import torch
from render import Renderer
import numpy as np


def project_and_render_point(renderer, point, temp_folder):
    """
    Project a 3D point onto different views from the camera, render the image,
    and store it in a temporary folder without using OpenCV.

    Args:
    - renderer: An instance of the Renderer class.
    - point: A 3D point as a list or tuple (x, y, z).
    - temp_folder: Path to the temporary folder where images will be stored.
    """
    point_3d = torch.tensor(
        [[point[0], point[1], point[2], 1.0]], dtype=torch.float32
    ).to(renderer.zero_tensor.device)
    for i, (view_mat, lightdir) in enumerate(
        zip(renderer.view_mats, renderer.lightdir)
    ):
        # Project the point onto this view
        point_2d = view_mat @ point_3d.t()
        x, y, z = point_2d[0, 0], point_2d[1, 0], point_2d[2, 0]

        # Normalize x, y to image coordinates
        x_img = (x * 0.5 + 0.5) * renderer.res
        y_img = (y * 0.5 + 0.5) * renderer.res

        # Render the model from this view
        img = renderer.target_imgs[i].cpu().numpy()[..., :3]

        fig, ax = plt.subplots()
        ax.imshow(img)
        circle = plt.Circle((x_img, y_img), 5, color="red", fill=True)
        ax.add_patch(circle)
        plt.axis("off")

        # Save the image to the temporary folder
        img_path = os.path.join(temp_folder, f"view_{i}.png")
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


# Example usage
point = (1, 2, 3)  # Example 3D point
renderer = Renderer(5, 512, fname="data/bunny.obj")
temp_folder = "./renders/temp/projection"
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
project_and_render_point(renderer, point, temp_folder)
print(f"Images saved in {temp_folder}")
