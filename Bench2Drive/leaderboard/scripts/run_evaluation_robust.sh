#!/bin/bash
# Must set CARLA_ROOT
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner

export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=1 # 1 -> output pictures
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

export MAX_ROUTE_TRIES=2
export ROUTE_TRIES=0
LAST_TRIED_ROUTE=""
MAX_RETRIES=9999
RETRY_COUNT=0
TIMEOUT=100

FIRST_GPU=$(echo ${GPU_RANK} | cut -d',' -f1)

get_current_route() {
    python3 -c "
import json, os
try:
    with open('$CHECKPOINT_ENDPOINT', 'r') as f:
        data = json.load(f)
    print(data['_checkpoint']['progress'][0])
except:
    print('')
"
}

cleanup() {
    echo "Cleaning up processes for GPU ${GPU_RANK}..."
    # 终止该 GPU 上的 CARLA 进程
    ps -ef | grep "graphicsadapter=${FIRST_GPU}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    # 终止相关端口的进程
    fuser -k ${PORT}/tcp 2>/dev/null
    fuser -k ${TM_PORT}/tcp 2>/dev/null
    sleep 10
}

run_evaluation() {
    echo -e "CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_robust.py --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=${DEBUG_CHALLENGE} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --traffic-manager-port=${TM_PORT} --gpu-rank=${FIRST_GPU} --timeout=${TIMEOUT}"
    CUDA_VISIBLE_DEVICES=${GPU_RANK} python "${LEADERBOARD_ROOT}"/leaderboard/leaderboard_evaluator_robust.py \
        --routes="${ROUTES}" \
        --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} \
        --checkpoint="${CHECKPOINT_ENDPOINT}" \
        --agent="${TEAM_AGENT}" \
        --agent-config="${TEAM_CONFIG}" \
        --debug=${DEBUG_CHALLENGE} \
        --record="${RECORD_PATH}" \
        --resume=${RESUME} \
        --port="${PORT}" \
        --traffic-manager-port="${TM_PORT}" \
        --gpu-rank="${FIRST_GPU}" \
        --timeout=${TIMEOUT} &
    local pid=$!
    wait $pid
    return $?
}

# 主循环：失败后自动重试
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "======================================================"
    echo "Try $((RETRY_COUNT + 1))/${MAX_RETRIES}, route try $(($ROUTE_TRIES + 1))/${MAX_ROUTE_TRIES} for GPU ${GPU_RANK}"
    echo "======================================================"
    
    run_evaluation
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Evaluation completed successfully on GPU ${GPU_RANK}!"
        exit 0
    fi

    # 线路计数器
    CURRENT_ROUTE=$(get_current_route)
    if [ "$CURRENT_ROUTE" = "$LAST_TRIED_ROUTE" ]; then
        export ROUTE_TRIES=$((ROUTE_TRIES + 1))
    else
        export ROUTE_TRIES=1
        export LAST_TRIED_ROUTE="$CURRENT_ROUTE"
    fi
    if [ $ROUTE_TRIES -gt $MAX_ROUTE_TRIES ]; then
        export ROUTE_TRIES=1
    fi

    # 总计数器
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Evaluation crashed with exit code ${EXIT_CODE}. Retry ${RETRY_COUNT}/${MAX_RETRIES}..."
    
    # 清理进程
    cleanup
    
    # 设置 RESUME 为 True 以从 checkpoint 恢复
    export RESUME=True
    
    sleep 5
done

echo "Max retries (${MAX_RETRIES}) reached for GPU ${GPU_RANK}. Exiting."
exit 1