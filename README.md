# N4DE: Neural 4D Evolution under Large Topological Changes from 2D Images

<div style="text-align: center">
<img src="static/Breaking Sphere.gif" width="600"/>
</div>

<br>

## Data
Please download the provided dataset and put them under the `data/` folder in root of the project. You can download the dataset from [xxxx]().
(**The dataset will be released and the link will be provided**)

## Setting up the environment
```bash
conda env create --name dynamic_nie --file=envs/main-env.yml
conda activate dynamic_nie

cd nvdiffrast/
pip install .
cd ../
```

Also, if you are using the __Gaussian Splatting__ Config, (for rendering head) you need to install the following:
```bash
# Inside root of the project

cd submodules/
pip install diff-gaussian-rasterization/
pip install simple-knn/
cd ../
```

Also, for using the HasGrid Encoder, you need to install `tiny-cuda-nn`:
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