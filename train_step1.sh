#!/usr/bin/env bash

# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
  exit 1
fi
# CONDA ENV
CONDA_NEW_ENV=taac2021-tagging
# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
  echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
  exit 1
fi
# CODE ROOT
CODE_ROOT=${JUPYTER_ROOT}/VideoStructuring
if [ ! -d "${CODE_ROOT}" ]; then
  echo "CODE_ROOT= ${CODE_ROOT}, not exists, exit"
  exit 1
fi
# DATASET ROOT
DATASET_ROOT=${CODE_ROOT}/dataset
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "DATASET_ROOT= ${DATASET_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "DATASET_ROOT= ${DATASET_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### activate conda env and check lib versions
# solve run problem in Jupyter Notebook
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
CONDA_CONFIG_FILE="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_CONFIG_FILE}" ]; then
  echo "CONDA_CONFIG_FILE= ${CONDA_CONFIG_FILE}, not exists, exit"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_CONFIG_FILE}"

# ###### activate conda env
# conda env by name
# conda activate ${CONDA_NEW_ENV}
# conda env by prefix
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
conda info --envs

# check tf versions
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
python -c "import tensorflow as tf; print(tf.__version__)"
# check np versions
python -c "import numpy as np; print(np.__version__)"
# check torch versions
python -c "import torch; print(torch.__version__)"

# #################### get 1st input argument as TYPE
TYPE=train
if [ -z "$1" ]; then
    echo "[Warning] TYPE is not set, using 'train' as default"
else
    TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "[Info] TYPE is ${TYPE}"
fi

# #################### execute according to TYPE
########## check
if [ "$TYPE" = "help" ]; then
  echo "Run for tagging: ./run.sh [TYPE] [Parameters]"
  echo "[TYPE] can be the following options:"
  echo "  ./run.sh help: help for ./run.sh"
  echo "  ./run.sh check: check conda environment"
  echo "  ./run.sh fix: fix conda environment when you restart from pausing"
  echo "  ./run.sh extract: feature extraction, no need for the baseline"
  echo "  ./run.sh gt: generate tagging gt files for training, no need for the baseline"
  echo "  ./run.sh train [CONFIG_FILE]: train with config file"
  echo "            CONFIG_FILE: optional, config file path, default is ${CODE_ROOT}/MultiModal-Tagging/configs/config.tagging.5k.yaml"
  echo "  ./run.sh test [CHECKPOINT_DIR] [OUTPUT_FILE_PATH] [TEST_VIDEOS_DIR] [TEST_VIDEOS_FEATS_DIR]"
  echo "            CHECKPOINT_DIR: relative model dir under ${CODE_ROOT}/MultiModal-Tagging/, such as 'checkpoints/tagging5k_temp/export/step_5000_0.7482'"
  echo "            OUTPUT_FILE_PATH: optional, relative output file path under ${CODE_ROOT}/MultiModal-Tagging/, default './results/tagging_5k_A.json'"
  echo "            TEST_VIDEOS_DIR: optional, test video directory, default is ${DATASET_ROOT}/videos/test_5k_A"
  echo "            TEST_VIDEOS_FEATS_DIR: optional, test video features directory, default is ${DATASET_ROOT}/tagging/tagging_dataset_test_5k"
  echo "  ./run.sh eval [RESULT_FILE_PATH] [GT_FILE_PATH]:"
  echo "            RESULT_FILE_PATH: result file for test data, such as './results/tagging_5k_A.json'"
  echo "            GT_FILE_PATH: gt file for test data, such as ${DATASET_ROOT}/tagging/test100.json"

  exit 0


########## train
elif [ "$TYPE" = "train" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd

  # ########## train
  echo "[Info] train with config= ${CONFIG_FILE}"
  time  python train_5fild.py
  time  python train.py  --config 'configs/step1/config1.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step1/config2.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step1/config3.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step1/config4.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step1/config5.tagging.5k.yaml'
  time  nohup python inference.py --model_pb './checkpoints/step1/tagging5k_temp1/export/' \
                            --output 'step1/results1/result_for_vis.txt' \
                            --output_json 'step1/results1/tagging_1.json' \
                            --tag_id_file '../dataset/label_id.txt' \
                            --test_dir '../dataset/videos/train_5k' \
                            --feat_dir  '../dataset/tagging/tagging_dataset_train_5k' &
                            
                            
  time  python inference.py --model_pb './checkpoints/step1/tagging5k_temp2/export/' \
                            --output 'step1/results2/result_for_vis.txt' \
                            --output_json 'step1/results2/tagging_2.json' \
                            --tag_id_file '../dataset/label_id.txt' \
                            --test_dir '../dataset/videos/train_5k' \
                            --feat_dir  '../dataset/tagging/tagging_dataset_train_5k'
                            
                            
  time  python inference.py --model_pb './checkpoints/step1/tagging5k_temp3/export/' \
                            --output 'step1/results3/result_for_vis.txt' \
                            --output_json 'step1/results3/tagging_3.json' \
                            --tag_id_file '../dataset/label_id.txt' \
                            --test_dir '../dataset/videos/train_5k' \
                            --feat_dir  '../dataset/tagging/tagging_dataset_train_5k'
                            
                            
  time  python inference.py --model_pb './checkpoints/step1/tagging5k_temp4/export/' \
                            --output 'step1/results4/result_for_vis.txt' \
                            --output_json 'step1/results4/tagging_4.json' \
                            --tag_id_file '../dataset/label_id.txt' \
                            --test_dir '../dataset/videos/train_5k' \
                            --feat_dir  '../dataset/tagging/tagging_dataset_train_5k'
                            
                            
  time  python inference.py --model_pb './checkpoints/step1/tagging5k_temp5/export/' \
                            --output 'step1/results5/result_for_vis.txt' \
                            --output_json 'step1/results5/tagging_5.json' \
                            --tag_id_file '../dataset/label_id.txt' \
                            --test_dir '../dataset/videos/train_5k' \
                            --feat_dir  '../dataset/tagging/tagging_dataset_train_5k'
  time python generate_soft_label.py
  
else
  echo "[Error] type= $TYPE not supported"

  exit 0
fi