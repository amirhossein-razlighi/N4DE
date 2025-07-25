#!/bin/bash
#SBATCH --job-name=SH_Training  # Job name
#SBATCH --partition=batch       # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=10      # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=8G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --output=./job_out/train_out.log      # Redirect stdout to a log file
#SBATCH --error=./job_out/train_err.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email
#SBATCH --gpus=a100-40g:1         # Request 1 GPU of gpu_model

srun -- mkenv -f /home/amirhossein_razlighi/codes/NIE_Animated/envs/main-env.yml -- \
/usr/bin/sh -c "export IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg" && \
export CC=/usr/bin/gcc-12 && \
export CXX=/usr/bin/g++-12 && \
export CUDA_HOME="/opt/modules/nvidia-cuda-12.3/" && \
export CUDA_VISIBLE_DEVICES=0 && \
python3 train_sh.py"