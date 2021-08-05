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
elif [ "$TYPE" = "check" ]; then
  echo "[Info] just check the conda env ${CONDA_NEW_ENV}"

  exit 0
elif [ "$TYPE" = "fix" ]; then
  echo "[Info] fix the environment when you restart from pausing"

  # #################### install system libraries
  if [ "${OS_ID}" == "ubuntu" ]; then
    echo "[Info] installing system libraries in ${OS_ID}"
    sudo apt-get update
    sudo apt-get install -y apt-utils
    sudo apt-get install -y libsndfile1-dev ffmpeg
  elif [ "${OS_ID}" == "centos" ]; then
    echo "[Info] installing system libraries in ${OS_ID}"
    yum install -y libsndfile libsndfile-devel ffmpeg ffmpeg-devel
  else
    echo "[Warning] os not supported for ${OS_ID}"
    exit 1
  fi

  # #################### recreate ipython kernels
  # conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
  # shellcheck source=/opt/conda/etc/profile.d/conda.sh
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"

  # add envs_dirs
  conda config --add envs_dirs ${JUPYTER_ROOT}/envs
  conda config --show | grep env

  # ###### create env and activate
  # TensorFlow 1.14 GPU dependencies - https://www.tensorflow.org/install/source#gpu
  # create env by prefix
  conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
  # create env by name
  # conda activate ${CONDA_NEW_ENV}
  conda info --envs
  conda info

  # create a kernel for conda env
  python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"

  exit 0
########## extract - extracted features are provided
elif [ "$TYPE" = "extract" ]; then
  DATASET_ROOT=${CODE_ROOT}/dataset
  echo "[Info] extracted features are provided, no need to execute TYPE= ${TYPE}"
  echo "  video features:            ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/video_npy/"
  echo "  audio features:            ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/audio_npy/"
  echo "  video banner image files:  ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/image_jpg/"
  echo "  text ocr and asr features: ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/text_txt/"

  exit 0
########## gt - generate gt files for training
elif [ "$TYPE" = "gt" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd

  # ########## 从原始视频（videos/train_5k_A）和标注信息（structuring/GourndTruth/train5k.txt, json格式）里生成标注文件（tagging_info.txt, csv格式）
  echo "[Info] generating tagging_info.txt for training data"
  OUTPUT_TAGGING_INFO_FILE=${DATASET_ROOT}/tagging/GroundTruth/tagging_info.txt
  if [ ! -f ${OUTPUT_TAGGING_INFO_FILE} ]; then
    echo "[Info] tagging_info.txt not exists, generating: OUTPUT_TAGGING_INFO_FILE= ${OUTPUT_TAGGING_INFO_FILE}"
    time python scripts/preprocess/json2info.py --video_dir ${DATASET_ROOT}/videos/train_5k_A \
                                                --json_path ${DATASET_ROOT}/structuring/GroundTruth/train5k.txt \
                                                --save_path ${OUTPUT_TAGGING_INFO_FILE} \
                                                --convert_type tagging
  else
    echo "[Info] tagging_info.txt exists, no need to generate: OUTPUT_TAGGING_INFO_FILE= ${OUTPUT_TAGGING_INFO_FILE}"
  fi

  # ########## 从标签字典文件、训练集标注文件（csv格式）、特征文件（tagging/tagging_dataset_train_5k/）等生成训练数据集
  echo "[Info] generating gt files for training data"
  OUTPUT_TRAIN_FILE=${DATASET_ROOT}/tagging/GroundTruth/datafile/train.txt
  OUTPUT_VAL_FILE=${DATASET_ROOT}/tagging/GroundTruth/datafile/val.txt
  if [[ ! -f ${OUTPUT_TRAIN_FILE} ]] || [[ ! -f ${OUTPUT_VAL_FILE} ]]; then
    rm -f ${OUTPUT_TRAIN_FILE} ${OUTPUT_VAL_FILE}
    echo "[Info] dataset files for training data not exists, generating: OUTPUT_TRAIN_FILE= ${OUTPUT_TRAIN_FILE}, OUTPUT_VAL_FILE= ${OUTPUT_VAL_FILE}"
    time python scripts/preprocess/generate_datafile.py --info_file ${DATASET_ROOT}/tagging/GroundTruth/tagging_info.txt \
                                                        --out_file_dir ${DATASET_ROOT}/tagging/GroundTruth/datafile/ \
                                                        --tag_dict_path ${DATASET_ROOT}/label_id.txt \
                                                        --frame_npy_folder ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/ \
                                                        --audio_npy_folder ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging/ \
                                                        --text_txt_folder ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/text_txt/tagging/ \
                                                        --image_folder ${DATASET_ROOT}/tagging/tagging_dataset_train_5k/image_jpg/tagging/
  else
    echo "[Info] dataset files for training data exists, no need to generate: OUTPUT_TRAIN_FILE= ${OUTPUT_TRAIN_FILE}, OUTPUT_VAL_FILE= ${OUTPUT_VAL_FILE}"
  fi

  exit 0
########## train
elif [ "$TYPE" = "train" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd



  # ########## train
  echo "[Info] train with config= ${CONFIG_FILE}"
  time  python train_5fild.py
  time  python train.py  --config 'configs/step2/config1.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step2/config2.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step2/config3.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step2/config4.tagging.5k.yaml' 
  time  python train.py  --config 'configs/step2/config5.tagging.5k.yaml'
  
else
  echo "[Error] type= $TYPE not supported"

  exit 0
fi