import pdb
import numpy as np
import torch
import torch.nn as nn
from utils import *
from grid_encoding import *
from tqdm import tqdm
from simple_knn._C import distCUDA2


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

    def __repr__(self):
        return self.__class__.__name__ + "(const={})".format(self.const)


class SIREN(nn.Module):
    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        self.use_hieararchial_consts = cfg.use_hierarchial_consts
        self.const = cfg.const
        if self.use_hieararchial_consts:
            self.consts = [cfg.const * (i + 1) for i in range(n_blocks + 1)]

        self.act = getattr(cfg, "act", Sine(self.const))

        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(dim, hidden_size))
        for i in range(n_blocks):
            lst = nn.Sequential()
            lst.append(nn.Linear(hidden_size, hidden_size))
            if self.use_hieararchial_consts:
                lst.append(Sine(self.consts[i]))
            else:
                lst.append(self.act)
            self.blocks.append(lst)

        self.blocks.append(nn.Linear(hidden_size, out_dim))

        # Initialization
        if isinstance(self.act, Sine):
            print("Sine activation function is used")
            self.apply(sine_init)
            if getattr(cfg, "not_first_layer_init", False):
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
        for block in self.blocks[:-1]:
            net = block(net)  # Activation function is also included in the block
        out = self.blocks[-1](net)
        return out


class EncoderPlusSDF(nn.Module):
    def __init__(
        self,
        in_features=3,
        device: str = "cuda:0",
    ):
        super(EncoderPlusSDF, self).__init__()

        F = 8
        L = 16
        self.encoder = get_a_grid_encoder(
            input_dims=3,
            F=F,
            L=L,
            log2_T=19,
            N_min=64,
            scale=1.5,
            interpolation="Linear",
        ).to(device)

        self.sdf_head = nn.Sequential(
            # nn.Linear(F * L + 64, 256),
            nn.Linear(F * L + 64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(128, 1),
            # nn.Linear(256, 1),
        ).to(device)

        self.num_time_freqs = 64
        self.num_pos_freqs = 64
        self.in_features = in_features

    def forward(self, coords):
        xyz = (coords[..., :3] + 1) / 2  # convert from [-1, 1] to [0, 1]
        time_posenc = positional_encoding(coords[..., 3:4], self.num_time_freqs)
        latent = self.encoder(xyz).float()
        encoded = torch.cat((latent, time_posenc), dim=-1)
        est_sdf = self.sdf_head(encoded)
        return est_sdf, latent

    def get_zero_points(
        self,
        t,
        mesh_res=32,
        offset=0,
        verbose=False,
        device="cuda:0",
        batch_size=20000,
        random=False,
    ):
        res = mesh_res
        bound = 1.0
        if not random:
            xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
            grid = np.concatenate(
                [ys[..., np.newaxis], xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1
            ).astype(np.float)
            grid = (grid / float(res - 1) - 0.5) * 2 * bound
            grid = grid.reshape(-1, 3)
        else:
            raise NotImplementedError("Random sampling is not implemented yet")

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
                xyz = (
                    torch.from_numpy(grid[sidx:eidx, :])
                    .float()
                    .to(device)
                    .view(1, -1, 3)
                )

                # concat time to 'xyz - offset' and feed in to the network
                inp = torch.cat(
                    (
                        xyz - offset,
                        torch.ones((xyz.shape[0], xyz.shape[1], 1)).to(device) * t,
                    ),
                    dim=2,
                ).squeeze(0)

                distances, _ = self.forward(inp)
                distances = distances.cpu().numpy()
            dists_lst.append(distances.reshape(-1))
        dists = np.concatenate([x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(
            -1
        )
        field = dists.reshape(res, res, res)
        try:
            vert, face, _, _ = skimage.measure.marching_cubes(
                field, level=0.0, spacing=[voxel_size] * 3, method="lorensen"
            )
        except:
            # print(inp)
            # print(inp.shape)
            # print(distances)
            # print(distances.shape)
            # print(field)
            # print(dists_lst)
            # print(dists)
            # print(res)
            print(
                f"If you want to see the trace of the error, uncomment the print statements in get_zero_points"
            )
            raise Exception("Failed to do marching cubes!")
        vert += voxel_origin
        vert -= offset
        return vert, face


class RenderHead(nn.Module):
    def __init__(self, sh_order=3, scaling_factor=2 / 200):
        super(RenderHead, self).__init__()
        self.sh_order = sh_order
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        F = 8
        L = 16

        self.grid_encoder = get_a_grid_encoder(
            3,
            L,
            F,
            interpolation="Linear",
        ).cuda()

        self.num_time_freqs = 64

        self.latent_head = nn.Sequential(
            nn.Linear(F * L + self.num_time_freqs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).cuda()

        self.sh_head = nn.Linear(64, 3 * (sh_order + 1) ** 2).cuda()
        self.opacity_head = nn.Linear(64, 1).cuda()
        # self.scaling_head = nn.Linear(64, 3).cuda()
        self.scaling_factor = scaling_factor
        self.rotation_head = nn.Linear(64, 4).cuda()  # quaternion

    def forward(self, x):
        xyz = (x[..., :3] + 1) / 2
        t_posenc = positional_encoding(x[..., 3:4], self.num_time_freqs)
        latent = self.grid_encoder(xyz).float()
        latent = self.latent_head(torch.cat((latent, t_posenc), dim=-1))

        sh_coeffs = self.sh_head(latent).view(-1, (self.sh_order + 1) ** 2, 3)
        opacity = self.opacity_activation(self.opacity_head(latent))
        # scaling = self.scaling_activation(self.scaling_head(latent))
        scaling = torch.full((x.shape[0], 3), self.scaling_factor).to(x.device)
        rotation = self.rotation_activation(self.rotation_head(latent))

        return sh_coeffs, opacity, scaling, rotation


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping for 3D points.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [n_points, 3],
     returns a tensor of size [n_points, mapping_size*2].
    """

    def __init__(self, num_input_channels=3, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 2, "Expected 2D input (got {}D input)".format(x.dim())

        n_points, channels = x.shape

        assert (
            channels == self._num_input_channels
        ), "Expected input to have {} channels (got {} channels)".format(
            self._num_input_channels, channels
        )

        x = x @ self._B.to(x.device)

        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
