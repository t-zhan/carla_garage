#!/bin/bash

MAX_RETRIES=10

ROUTES="leaderboard/data/routes_validation.xml"
AGENT="team_code/sensor_agent.py"
AGENT_CONFIG="pretrained_models/town13_withheld"
CHECKPOINT="${AGENT_CONFIG}/simulation_results.json"
TIMEOUT=50

# # 初始化清空 checkpoint 文件
# > "${CHECKPOINT}"

cleanup() {
    killall -9 -r CarlaUE4-Linux 2>/dev/null
    pkill -9 -f "carla-rpc-port" 2>/dev/null
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null
    sleep 5
}

start_carla() {
    carla/CarlaUE4.sh -RenderOffScreen > /dev/null 2>&1 &
    sleep 30
}

for i in $(seq 1 $MAX_RETRIES); do
    cleanup
    start_carla
    
    python leaderboard/leaderboard/leaderboard_evaluator_local.py \
        --routes "${ROUTES}" \
        --agent "${AGENT}" \
        --agent-config "${AGENT_CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --resume 1 \
        --timeout ${TIMEOUT}
    
    [ $? -eq 0 ] && exit 0
    echo "Retry $i/$MAX_RETRIES..."
done

exit 1