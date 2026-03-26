# 实际执行删除命令（保留时间戳最大的）
find Bench2Drive/leaderboard/data/eval_bench2drive220_max_traj -maxdepth 1 -type d -name "RouteScenario_*" -print0 | while IFS= read -r -d $'\0' dir; do
    base=$(basename "$dir")
    awk -F '_' -v base="$base" -v path="$dir" '{
        prefix = ""
        for (i=1; i<=(NF-7); i++) prefix = prefix (i>1?"_":"") $i
        timestamp = $(NF-6)"_"$(NF-5)"_"$(NF-4)"_"$(NF-3)"_"$(NF-2)"_"$(NF-1)"_"$NF
        print prefix, timestamp, base, path
    }' <<< "$base"
done | sort -k1,1 -k2r | awk '{
    if (last_prefix == $1) {
        print "rm -rf \"" $4 "\""
        # system("rm -rf \"" $4 "\"")  # 实际执行删除!!!!!
    } 
    # else {
    #     if (NR > 1) print "# 保留目录: " prev_dir
    # }
    prev_dir = $4
    last_prefix = $1
}
# END { if (NR > 0) print "# 保留目录: " prev_dir }'