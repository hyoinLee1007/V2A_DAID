#!/usr/bin/env bash
# lambda_c 민감도 실험. DAID_baseline 과 동일한 가중치·MPPI 설정·seed 를 두고
# 접촉 항의 계수만 바꾼다. 조건 사이에 달라지는 것은 그 한 줄뿐이다.
#
#   ./run_lambda_sweep.sh                 # 0 0.1 0.5 1.0 2.0
#   ./run_lambda_sweep.sh 0.5 1.0         # 일부만
set -e
cd "$(dirname "$0")"

TASK=cupmove
RAW=../reconstruction/dataset/cupmove
SIDE=left
TPL=/tmp/exp_base.yaml
CFG=config/override/do_as_i_do.yaml
OUT=outputs/sharpa/$SIDE/$TASK/0
LOG=outputs/sweep_logs
LAMBDAS=("${@:-0 0.1 0.5 1.0 2.0}")
read -ra LAMBDAS <<< "${LAMBDAS[@]}"

[ -f "$TPL" ] || { echo "템플릿이 없습니다: $TPL"; exit 1; }
mkdir -p "$LOG"
cp "$CFG" "$CFG.sweep_backup"
trap 'cp "$CFG.sweep_backup" "$CFG"; echo "config 복원됨"' EXIT

for L in "${LAMBDAS[@]}"; do
    DST=outputs/sharpa/$SIDE/${TASK}_lam${L}
    echo "=============== lambda_c = $L ==============="
    sed "s/^robot_contact_attract_scale: LAMBDA_C/robot_contact_attract_scale: $L/" "$TPL" > "$CFG"
    grep -E "^robot_contact_attract_scale|^robot_contact_repel_scale" "$CFG"

    conda run --no-capture-output -n retargeting python launch.py \
        --task "$TASK" --raw-dir "$RAW" --force --no-wait-on-finish \
        > "$LOG/lam${L}.log" 2>&1

    # 완료 확인: 이 세 가지가 다 맞아야 결과로 취급한다. 예전에 타임아웃으로
    # 죽은 실행의 이전 결과물을 새 결과로 착각한 적이 있다.
    if ! tr '\r' '\n' < "$LOG/lam${L}.log" | grep -q "Final object tracking error"; then
        echo "  [!] lambda=$L 실행이 완료되지 않았습니다 — 건너뜁니다"
        continue
    fi
    rm -rf "$DST"; mkdir -p "$DST"; cp -r "$OUT" "$DST/"
    tr '\r' '\n' < "$LOG/lam${L}.log" | grep "Final object tracking error" | tail -1
    echo "  -> $DST"
done

echo
echo "=============== 지표 ==============="
for L in "${LAMBDAS[@]}"; do
    D=outputs/sharpa/$SIDE/${TASK}_lam${L}/0
    [ -f "$D/trajectory_mjwp.npz" ] || continue
    conda run -n retargeting python contact_metrics.py --run "$D" --side "$SIDE" \
        --heatmap ../reconstruction/heatmap_out/contact_heatmap_palmar.npz \
        --map outputs/robot_contact_map_left.npz --label "lambda_c = $L"
done
