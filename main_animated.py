import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from models_animated import *
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main(config):
    if config.with_texture:
        from train_with_splats import perform_training
    else:
        from train import perform_training

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderPlusSDF().to(device)
    render_model = RenderHead(sh_order=3, scaling_factor=2 / config.mesh_res_base).to(
        device
    )

    if os.path.exists("checkpoints/model.pth") and not config._continue:
        model.load_state_dict(torch.load("checkpoints/model.pth"))
        render_model.load_state_dict(torch.load("checkpoints/render_init.pth"))
        print("Loaded model from initial checkpoint folder: {}".format("checkpoints/"))

    logger = SummaryWriter(log_dir=config.expdir, flush_secs=5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.99),
        eps=1e-15,
    )

    if config.num_frames is None:
        config.num_frames = 1

    if config._continue:
        qbar = tqdm(range(config.last_epoch, config.epochs))
        model.load_state_dict(
            torch.load(f"{config.expdir}/checkpoints/model_{config.last_epoch}.pth")
        )
        if config.with_texture:
            render_model.load_state_dict(
                torch.load(
                    f"{config.expdir}/checkpoints/render_head_{config.last_epoch}.pth"
                )
            )

        optimizer.load_state_dict(
            torch.load(
                f"{config.expdir}/checkpoints/sdf_optimizer_{config.last_epoch}.pth"
            )
        )
        print("Loaded model from checkpoints in {}".format(config.folder_name))
    else:
        qbar = tqdm(range(config.epochs))

    best_loss = np.inf

    gt_sdf = torch.zeros(config.max_v, 1).to(device)
    F = torch.zeros(config.max_v, 1).to(device)
    vertices = torch.zeros((config.max_v, 4)).to(device)  # last element is time
    normals = torch.zeros((config.max_v, 4)).to(device)
    faces = torch.empty((config.max_v, 3), dtype=torch.int32).to(device)
    vertices.requires_grad_()

    if config.use_amp:
        print("Using Automatic Mixed Precision")

    os.makedirs(f"{config.expdir}/checkpoints", exist_ok=True)

    perform_training(
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
        prof_=None,
    )


if __name__ == "__main__":
    config = parse_config(create_dir=True, consider_max_dir=True)
    main(config)
