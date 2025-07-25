#!/bin/bash
#SBATCH --job-name=Deformable_Reconstruction
#SBATCH --partition=batch       # Specify the partition (debug|batch)
#SBATCH --nodes=1
#SBATCH --nodelist=gcp-us-0,gcp-us-1,gcp-us-2
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=10       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=2G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --output=./job_out/train_out_%j.log
#SBATCH --error=./job_out/train_err_%j.error
#SBATCH --mail-type=ALL
#SBATCH --gpus=a100-40g:1       # a100-40g | l4-24g

srun -- mkenv -f /home/amirhossein_razlighi/codes/NIE_Animated/envs/main-env.yml -- \
    /usr/bin/sh -c "export IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg' && \
export CC=/usr/bin/gcc-12 && \
export CXX=/usr/bin/g++-12 && \
export CUDA_HOME='/opt/modules/nvidia-cuda-12.3.1/' && \
PACKAGE_NAME='nvdiffrast' && \
if ! pip show \$PACKAGE_NAME > /dev/null 2>&1; then \
    echo 'nvdiffrast is not installed. Installing...' && \
    cd nvdiffrast/ && \
    pip install . && \
    cd ..; \
else \
    echo 'nvdiffrast is already installed. Skipping installation.'; \
fi && \
PACKAGE_NAME='tinycudann' && \
if ! pip show \$PACKAGE_NAME > /dev/null 2>&1; then \
    echo 'tiny-cuda is not installed. Installing...' && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch; \
else \
    echo 'tiny-cuda is already installed. Skipping installation.'; \
fi && \
pip install submodules/diff-gaussian-rasterization && \
pip install submodules/simple-knn && \
python3 main_animated.py --config /home/amirhossein_razlighi/codes/NIE_Animated/config/deformable_chair.yaml"
