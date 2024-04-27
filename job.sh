#!/bin/bash
#SBATCH --job-name=NIE
#SBATCH --partition=debug       # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=03:15:00         # Job timeout
#SBATCH --output=./job_out/myjob.log      # Redirect stdout to a log file
#SBATCH --error=./job_out/myjob.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email
#SBATCH --gpus=l4-24g:2      # Request 2 GPUs of model L4

srun -- mkenv -f /home/amirhossein_razlighi/codes/NIE_Animated/test-env.yml -- \
bash -c "cd nvdiffrast/ && \
pip install . && \
cd .. && \
export CC=/usr/bin/gcc-12 && \
export CXX=/usr/bin/g++-12 && \
export CUDA_HOME="/opt/nvidia-cuda-12.3/" && \
python3 main_animated.py --config ./config/animation_deform.yaml"