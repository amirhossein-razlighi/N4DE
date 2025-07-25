
# N4DE: Neural 4D Evolution under Large Topological Changes from 2D Images
Official PyTorch implementation of ["N4DE: Neural 4D Evolution under Large Topological Changes from 2D Images"](https://arxiv.org/pdf/2411.15018) 

By [AmirHossein Razlighi](https://scholar.google.com/citations?user=JbQgt-QAAAAJ&hl=en), [Tiago Novello]() , [Asen Nachkov](), [Thomas Probst](), [Danda Paudel]()

In the literature, it has been shown that the evolution of the known explicit 3D surface to the target one can be learned from 2D images using the instantaneous flow field, where the known and target 3D surfaces may largely differ in topology. We are interested in capturing 4D shapes whose topology changes largely over time. We encounter that the straightforward extension of the existing 3D-based method to the desired 4D case performs poorly.
In this work, we address the challenges in extending 3D neural evolution to 4D under large topological changes by proposing two novel modifications. More precisely, we introduce (i) a new architecture to discretize and encode the deformation and learn the SDF and (ii) a technique to impose the temporal consistency. (iii) Also, we propose a rendering scheme for color prediction based on Gaussian splatting. Furthermore, to facilitate learning directly from 2D images, we propose a learning framework that can disentangle the geometry and appearance from RGB images. This method of disentanglement, while also useful for the 4D evolution problem that we are concentrating on, is also novel and valid for static scenes. Our extensive experiments on various data provide awesome results and, most importantly, open a new approach toward reconstructing challenging scenes with significant topological changes and deformations.

<div style="text-align: center">
<img src="static/Breaking Sphere.gif" width="600"/>
</div>

<br>

## Data
Please download the provided dataset and put them under the `data/` folder in root of the project. You can download the dataset from [xxxx]().

## Setting up the environment
```bash
conda env create --name dynamic_nie --file=envs/main-env.yml
conda activate dynamic_nie

cd submodules/nvdiffrast/
pip install .
cd ../../
```

Also, if you are using the __Gaussian Splatting__ Config, (for rendering head) you need to install the following:
```bash
# Inside root of the project

cd submodules/
pip install diff-gaussian-rasterization/
pip install simple-knn/
cd ../
```

Also, for using the HashGrid Encoder, you need to install `tiny-cuda-nn`:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch;
```

## How to initialize the SDF Module to a sphere
```
python3 sphere_init.py
```
This will save a checkpoint to `checkpoints/model.pth`. It will be loaded by default via the model when you run `main_animated.py`.

## How to initialize the Rendering Module to fit splats on a sphere
```
python3 splat_init.py
```
This will save a checkpoint to `checkpoints/render_init.pth`. It will be loaded by default via the model when you run `main_animated.py`.

## How to train the model and fit to a scene
```
python3 main_animated.py --config config/{Your desired config}.yaml
```
You can go through the `configs/` and see some of the pre-saved configs that we mentioned in the paper.

## Trace the process on tensorboard
```
tensorboard --logdir=./logs --bind_all
```

## Inference
You can see sample inferences of the model in the notebooks provided in `notebooks/` folder.