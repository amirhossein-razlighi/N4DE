import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np

try:
    import cv2 as cv
except:
    print("OpenCV not found. Some functionality may not work.")


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir  # E.g. data/Basketball/
        self.json_data = json.load(open(os.path.join(root_dir, "train_meta.json")))
        self.fn = self.json_data["fn"]
        self.w2c = self.json_data["w2c"]
        self.intrinsic = self.json_data["k"]

    def __getitem__(self, frame_idx):
        imgs = [
            cv.imread(os.path.join(self.root_dir, "ims", f"{fn}"))
            for fn in self.fn[frame_idx]
        ]
        segs = [
            cv.imread(
                os.path.join(self.root_dir, "seg", f"{fn[:-3]}" + "png"),
                cv.IMREAD_GRAYSCALE,
            )
            for fn in self.fn[frame_idx]
        ]
        w2cs = self.w2c[frame_idx]
        intrinsics = self.intrinsic[frame_idx]
        return imgs, segs, w2cs, intrinsics

    def __len__(self):
        return len(self.fn)


class MovingPersonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.frame_paths = [
            f for f in os.listdir(os.path.join(root_dir, "images")) if f != ".DS_Store"
        ]
        self.num_frames = len(self.frame_paths)

    def __getitem__(self, frame_idx):
        data = json.load(
            open(
                os.path.join(
                    self.root_dir,
                    "train",
                    f"transform_{self.frame_paths[frame_idx]}.json",
                )
            )
        )
        views = data["frames"]
        from_na = data["from_na"]
        scale = data["scale"]
        offset = data["offset"]
        imgs = []
        c2ws = []
        intrinsics = []

        for view in views:
            img = cv.imread(
                os.path.join(self.root_dir, view["file_path"][3:]),
                cv.IMREAD_COLOR,
            )
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # the background is white. We make it black to be consistent with our renders
            img[img == 255] = 0
            imgs.append(img)
            c2ws.append(view["transform_matrix"])
            intrinsics.append(view["intrinsic_matrix"])

        return (
            torch.stack([torch.tensor(img) for img in imgs]) / 255.0,
            torch.tensor(c2ws),
            torch.tensor(intrinsics),
            from_na,
            torch.tensor(scale),
            torch.tensor(offset),
        )

    def __len__(self):
        return self.num_frames


if __name__ == "__main__":
    from render import DatasetRenderer
    import matplotlib.pyplot as plt

    dataset = MovingPersonDataset("data/MovingPerson")
    imgs, c2ws, intrinsics = dataset[0]
    renderer = DatasetRenderer(imgs, c2ws, intrinsics)
    plt.imsave("temp.png", renderer.target_imgs[0].cpu().numpy())
