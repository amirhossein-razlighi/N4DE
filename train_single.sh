#!/bin/bash
#SBATCH --job-name=NIE_Train    # Job name
#SBATCH --partition=batch       # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=12:00:00         # Job timeout
#SBATCH --output=./job_out/train_out.log      # Redirect stdout to a log file
#SBATCH --error=./job_out/train_err.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email
#SBATCH --gpus=l4-24g:1         # Request 1 GPU of model L4

srun -- mkenv -f /home/amirhossein_razlighi/codes/NIE_Animated/test-env.yml -- \
/usr/bin/sh -c "cd nvdiffrast/ && \
pip install . && \
cd .. && \
export IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg" && \
python3 main_animated.py --config ./config/animation_deform.yaml"

# export CC=/usr/bin/gcc-12 && \
# export CXX=/usr/bin/g++-12 && \
# export CUDA_HOME="/opt/modules/nvidia-cuda-12.3/" && \