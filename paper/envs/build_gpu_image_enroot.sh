#!/bin/bash
#SBATCH --job-name=mlr3torch-gpu-image
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=build-gpu-image-%j.out

# Builds paper/envs/Dockerfile.linux.gpu with enroot (no docker on the cluster): the Dockerfile is translated by
# dockerfile2sh.py into build.sh, which is executed in a container created from the Dockerfile's base image.
# Runs on the smallest MIG slice, because the CPU partition (lrz-cpu) had a queue of more than a week.
# Before submitting, create the build directory $DIR/gpubuild with build.sh and context/renv.lock:
#   python3 paper/envs/dockerfile2sh.py paper/envs/Dockerfile.linux.gpu > $DIR/gpubuild/build.sh
#   mkdir -p $DIR/gpubuild/context && cp paper/renv.lock $DIR/gpubuild/context/
set -e
DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru48nas2/mlr3torch-paper-review
TMP=/tmp/$USER-mlr3torch-gpu-build
export ENROOT_DATA_PATH=$TMP/data ENROOT_CACHE_PATH=$TMP/cache ENROOT_TEMP_PATH=$TMP/tmp
mkdir -p $ENROOT_DATA_PATH $ENROOT_CACHE_PATH $ENROOT_TEMP_PATH
df -h /tmp
trap 'rm -rf $TMP' EXIT

enroot import -o $TMP/base.sqsh 'docker://nvcr.io#nvidia/cuda:12.6.1-devel-ubuntu22.04'
enroot create --name mlr3torch-gpu-build $TMP/base.sqsh
enroot start --root --rw --mount $DIR/gpubuild:/build mlr3torch-gpu-build bash /build/build.sh
enroot export -o $DIR/mlr3torch-jss+gpu.sqsh mlr3torch-gpu-build
ls -la $DIR/mlr3torch-jss+gpu.sqsh
echo IMAGE_DONE
