#!/bin/bash
#SBATCH --job-name=b2d_009
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/b2d_009_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/b2d_009_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=2080-galvani

export CARLA_ROOT=/home/autodrive/Projects/carla_garage/carla
export WORK_DIR=/home/autodrive/Projects/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:/home/autodrive/Projects/carla_garage/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
# BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220
ROUTE_DIR=${WORK_DIR}/leaderboard/data/route_splits/bench2drive220
LOG_DIR=${WORK_DIR}/leaderboard/data/logs/bench2drive220
TEAM_AGENT=/home/autodrive/Projects/carla_garage/team_code/sensor_agent_max.py
# Must set YOUR_CKPT_PATH
# MODEL_NAME_OR_PATH=/home/autodrive/Projects/carla_garage/pretrained_models/MaxAR_MiMo-VL-7B-RL_20250917005842/epoch_4
TEAM_CONFIG=/home/autodrive/Projects/carla_garage/pretrained_models/all_towns  # /home/autodrive/Projects/carla_garage/models/max_v1
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=max  # tfpp
SAVE_PATH=${WORK_DIR}/leaderboard/data/eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Example, 8*H100, 1 task per gpu
# GPU_RANK_LIST=(0 1)
GPU_RANK_LIST=("0,1,2,3") # ("0,1" "2,3")
TASK_LIST=(0)
TASK_NUM=${#TASK_LIST[@]}

mkdir -p $(dirname ${ROUTE_DIR})
mkdir -p $(dirname ${LOG_DIR})

if [ ! -d "${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP}" ]; then
    mkdir -p "${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP}"
    echo -e "\033[32m Directory ${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP} created. \033[0m"
else
    echo -e "\033[32m Directory ${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${ROUTE_DIR}_${ALGO}_${PLANNER_TYPE}_${TASK_NUM}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    # TASK_NUM=8 # 8*H100, 1 task per gpu
    python ${WORK_DIR}/tools/split_xml.py $ROUTE_DIR $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${ROUTE_DIR}_${ALGO}_${PLANNER_TYPE}_${TASK_NUM}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
echo -e "\033[32m GPU_RANK_LIST: ${GPU_RANK_LIST[@]} \033[0m"
echo -e "\033[32m TASK_LIST: ${TASK_LIST[@]} \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
      PORT=$((BASE_PORT + i * 150))
      TM_PORT=$((BASE_TM_PORT + i * 150))
      ROUTES="${ROUTE_DIR}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
      CHECKPOINT_ENDPOINT="${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
      mkdir -p "${WORK_DIR}/checkpoint_endpoint/${ALGO}_b2d_${PLANNER_TYPE}_${TIMESTAMP}"
      GPU_RANK=${GPU_RANK_LIST[$i]}
      echo -e "\033[32m ALGO: $ALGO \033[0m"
      echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
      echo -e "\033[32m TASK_ID: $i \033[0m"
      echo -e "\033[32m PORT: $PORT \033[0m"
      echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
      echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
      echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
      echo -e "\033[32m bash ${WORK_DIR}/leaderboard/scripts/run_evaluation_with_recovery_max.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
      echo -e "***********************************************************************************"
      bash ${WORK_DIR}/leaderboard/scripts/run_evaluation_with_recovery_max.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${LOG_DIR}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}_${TIMESTAMP}.log &
      sleep 5
done
wait