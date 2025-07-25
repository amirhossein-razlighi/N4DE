#!/bin/bash
#SBATCH --job-name=RenderTest   # Job name
#SBATCH --partition=debug       # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=4       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=00:15:00         # Job timeout
#SBATCH --output=./job_out/render_out.log      # Redirect stdout to a log file
#SBATCH --error=./job_out/render_err.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email
#SBATCH --gpus=l4-24g:1      # Request 1 GPU of model L4

srun -- mkenv -f /home/amirhossein_razlighi/codes/NIE_Animated/test-env.yml -- \
/usr/bin/sh -c \
"export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg && \
cd nvdiffrast/ && \
pip install . && \
cd .. && \
python3 render_a_checkpoint.py --num_frames 3 --checkpoint /home/amirhossein_razlighi/codes/NIE_Animated/logs/9_Animation1/best.ckpt --output_dir /home/amirhossein_razlighi/codes/NIE_Animated/renders/Intermediate_4/ --mesh_dir /home/amirhossein_razlighi/codes/NIE_Animated/data/Animation_1"

# python3 render_a_checkpoint.py --num_frames 3 --checkpoint /home/amirhossein_razlighi/codes/NIE_Animated/logs/8_Chair_Deform_Animation_Complete/iter_0002900.ckpt --output_dir /home/amirhossein_razlighi/codes/NIE_Animated/renders/Chair_deform/ --mesh_dir /home/amirhossein_razlighi/codes/NIE_Animated/data/Chair_Deform
# python3 render.py
# export CC=/usr/bin/gcc-12 && \
# export CXX=/usr/bin/g++-12 && \
# export CUDA_HOME="/opt/modules/nvidia-cuda-12.3/" && \