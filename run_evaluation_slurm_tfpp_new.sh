#!/bin/bash
#SBATCH --job-name=eval_server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.err
#SBATCH --partition=2080-galvani

BENCHMARK="routes_validation"
MODEL_DIR="/home/autodrive/Projects/carla_garage/pretrained_models/town13_withheld"
CODE_ROOT="/home/autodrive/Projects/carla_garage"
CARLA_ROOT="/home/autodrive/Projects/carla_garage/carla"
PARTITION="garage_2"
USERNAME="autodrive"
TEAM_CODE="team_code"
EPOCHS="model_0030"
NUM_REPETITIONS=3

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

for i in $(seq 1 3); do
  ex_name=$(printf "tfpp_009_%01d" "$((i - 1))")
  python -u evaluate_routes_slurm_tfpp.py \
    --experiment "${ex_name}" \
    --benchmark "${BENCHMARK}" \
    --model_dir "${MODEL_DIR}" \
    --code_root "${CODE_ROOT}" \
    --carla_root "${CARLA_ROOT}" \
    --partition "${PARTITION}" \
    --username "${USERNAME}" \
    --team_code "${TEAM_CODE}" \
    --epochs "${EPOCHS}" \
    --num_repetitions ${NUM_REPETITIONS} &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
