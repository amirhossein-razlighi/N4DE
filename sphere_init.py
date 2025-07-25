import torch
import torch.nn as nn
from models_animated import *
from utils import *
from tqdm import tqdm
from loss import *
import matplotlib.pyplot as plt
import os


def sphere_sdf(x, y, z, radius=1.0):
    return torch.sqrt(x**2 + y**2 + z**2) - radius


def ellipsoid_sdf(x, y, z, a=1.0, b=1.5, c=2.0):
    return torch.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2) - 1.0


def cube_sdf(x, y, z, size=1.0):
    return torch.max(
        torch.max(torch.abs(x) - size, torch.abs(y) - size), torch.abs(z) - size
    )


# Generate Sample Points
def generate_samples(num_samples, shape="sphere"):
    points = torch.rand(num_samples, 3) * 2 - 1  # Random points in [-1, 1]^3
    if shape == "sphere":
        sdf_values = sphere_sdf(points[:, 0], points[:, 1], points[:, 2])
    elif shape == "ellipsoid":
        sdf_values = ellipsoid_sdf(points[:, 0], points[:, 1], points[:, 2])
    elif shape == "cube":
        sdf_values = cube_sdf(points[:, 0], points[:, 1], points[:, 2])
    else:
        raise ValueError("Invalid shape")
    return points, sdf_values


def main(use_pre_trained=False, name="model"):
    # torch.manual_seed(2024)
    device = torch.device("cuda")
    model = EncoderPlusSDF().to(device)
    if use_pre_trained:
        model.load_state_dict(torch.load(f"checkpoints/{name}.pth"))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.99),
        eps=1e-15,
    )

    initialization_type = "sphere"  # sphere | ellipsoid | cube
    num_samples = 2**18
    batch_size = 2**14
    loss_function = nn.MSELoss()
    mape_loss_fn = mape_loss
    best_loss = torch.inf
    num_epochs = 500
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        for _ in range(0, num_samples, batch_size):
            optimizer.zero_grad()
            points, correct_sdf = generate_samples(
                batch_size, shape=initialization_type
            )
            points, correct_sdf = points.to(device), correct_sdf.to(device)
            time = torch.rand(batch_size, 1).to(device)
            points = torch.cat([points, time], dim=1)
            points.requires_grad_(True)

            predicted_sdf, _ = model(points)
            loss = loss_function(
                predicted_sdf.squeeze(), correct_sdf
            ) + 0.2 * mape_loss_fn(predicted_sdf.squeeze(), correct_sdf)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (num_samples // batch_size)
        pbar.set_postfix(
            {
                "Avg Loss": avg_epoch_loss,
                "Best Loss": best_loss if best_loss != torch.inf else "N/A",
            }
        )
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), f"checkpoints/{name}.pth")


if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if os.path.exists("checkpoints/model.pth"):
        print("Model already exists. Overwrite? (y/n/exit/suggest)")
        response = input()
        if response == "y":
            print("Overwriting model. Training from scratch.")
            main()
        elif response == "n":
            print("Using current model as a pre-trained model.")
            main(use_pre_trained=True)
        elif response == "suggest":
            name = input("Enter a name for the model: ")
            print(f"Using {name} as the model name.")
            main(name=name)
    else:
        main()
