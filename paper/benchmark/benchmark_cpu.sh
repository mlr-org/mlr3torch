#!/bin/bash
#SBATCH --job-name=mlr3torch-benchmark-cpu
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=06:00:00
#SBATCH --output=mlr3torch-benchmark-cpu-%j.out

# CPU benchmark on the LRZ Linux Cluster (CoolMUC-4), using the CPU docker image converted to a squashfs image.
# The whole node (2 x 56 cores, 224 logical CPUs) is requested so that no other jobs interfere with the time measurements.
# The benchmark itself sets the number of threads (1 or 16) via torch_set_num_threads().

module load slurm_setup
module load apptainer/1.3.4

DIR=$SCRATCH/mlr3torch-paper-review
cd $DIR

echo "Node: $(hostname)"
lscpu
free -g

apptainer exec --containall \
  --bind $DIR:/mnt/data \
  --env PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --env LANG=en_US.UTF-8 \
  --env TZ=Etc/UTC \
  mlr3torch-jss+cpu.sqsh bash -c "
  cd /mnt/data/paper
  Rscript benchmark/linux-cpu.R
"
