import pdb
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from utils import *


# Initialization functions are borrowed from:
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L622
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, const=30.0):
        super().__init__()
        self.const = const

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.const * input)


class Net(nn.Module):
    """Decoder conditioned by adding.

    Example configuration:
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """

    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(dim, hidden_size))
        for _ in range(n_blocks):
            self.blocks.append(nn.Linear(hidden_size, hidden_size))
        self.blocks.append(nn.Linear(hidden_size, out_dim))
        self.act = Sine(cfg.const)

        # Initialization
        self.apply(sine_init)
        self.blocks[0].apply(first_layer_sine_init)
        if getattr(cfg, "zero_init_last_layer", False):
            torch.nn.init.constant_(self.blocks[-1].weight, 0)
            torch.nn.init.constant_(self.blocks[-1].bias, 0)

    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        net = x  # (bs, n_points, dim)
        self.debug_res = []
        for block in self.blocks[:-1]:
            net = self.act(block(net))
            self.debug_res.append(net)
        out = self.blocks[-1](net)
        self.debug_res.append(out)
        return out


class SDFModule(LightningModule):
    def __init__(self, in_features=3, w0_initial=30.0, cfg=None, f: str = None):
        super().__init__()
        self.synthesis_nw = Net("", cfg)
        if f is not None:
            # Uncomment if you need to load from 'sphere'
            state_dict = torch.load(f)["net"]
            # randomly initialize the time weights to be between 0 and 1 and add them to state_dict
            # which is [512, 3] so it become [512, 4]
            # state_dict[f"blocks.0.weight"] = torch.cat(
            #     (state_dict[f"blocks.0.weight"], torch.rand(512, 1).cuda()), dim=1
            # )
            self.synthesis_nw.load_state_dict(state_dict)

            # In case you need to load from a specific checkpoint
            # state_dict = torch.load(f)
            # new_state_dict = {}
            # for key in state_dict.keys():
            #     new_state_dict[key[13:]] = state_dict[key]
            # self.synthesis_nw.load_state_dict(new_state_dict)

        self.in_features = in_features

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)

    def forward(self, coords):
        est_sdf = self.synthesis_nw(coords)
        return est_sdf

    def get_zero_points(self, t, extent=10, mesh_res=32, offset=0, verbose=False):
        res = mesh_res
        bound = 1.0
        batch_size = 10000
        xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
        grid = np.concatenate(
            [ys[..., np.newaxis], xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1
        ).astype(np.float)
        grid = (grid / float(res - 1) - 0.5) * 2 * bound
        grid = grid.reshape(-1, 3)
        voxel_size = 2.0 / (res - 1)
        voxel_origin = -1 * bound

        dists_lst = []
        pbar = range(0, grid.shape[0], batch_size)
        if verbose:
            pbar = tqdm.tqdm(pbar)
        for i in pbar:
            sidx, eidx = i, i + batch_size
            eidx = min(grid.shape[0], eidx)
            with torch.no_grad():
                xyz = torch.from_numpy(grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
                # concat time to 'xyz - offset' and feed in to the network
                # inp = torch.cat(
                #     (
                #         xyz - offset,
                #         torch.ones((xyz.shape[0], xyz.shape[1], 1)).cuda() * t,
                #     ),
                #     dim=2,
                # )
                distances = self.forward(xyz)
                distances = distances.cpu().numpy()
            dists_lst.append(distances.reshape(-1))
        dists = np.concatenate([x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)

        field = dists.reshape(res, res, res)
        try:
            vert, face, _, _ = skimage.measure.marching_cubes(
                field, level=0.0, spacing=[voxel_size] * 3, method="lorensen"
            )
        except:
            for item in self.synthesis_nw.debug_res:
                print(item)
            print(inp)
            print(inp.shape)
            print(distances)
            print(distances.shape)
            print(field)
            print(dists_lst)
            print(dists)
            print(res)
            raise Exception("Failed to do marching cubes!")
        vert += voxel_origin
        vert -= offset
        return vert, face
