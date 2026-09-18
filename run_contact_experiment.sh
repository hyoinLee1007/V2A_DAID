#!/usr/bin/env bash
# 한 태스크에 대해 접촉 맵을 만들고, 접촉 항만 켜고 끈 두 조건을 비교한다.
#
# 비교가 귀속 가능하도록 두 조건은 lambda_c 한 줄만 다르다. 같은 재구성, 같은
# 레퍼런스(ray 보정 포함), 같은 MPPI 설정, 같은 seed, 같은 기존 가중치를 쓴다.
# ray 보정은 접촉 맵의 전제조건이라 lambda_c = 0 조건에도 켜 둔다 — 안 그러면
# 두 조건의 레퍼런스가 달라져서 차이가 접촉 항 때문인지 알 수 없다.
#
#   ./run_contact_experiment.sh lotion_index
#   ./run_contact_experiment.sh lotion_index "0 0.5 1.0 2.0"     # 민감도 스윕
set -e
ROOT=/home/intern/do-as-i-do
REC=$ROOT/reconstruction
RET=$ROOT/retargeting
CONDA=/home/intern/miniconda3/bin/conda
PY=/home/intern/miniconda3/envs/retargeting/bin/python

TASK=${1:?태스크 이름이 필요합니다}
read -ra LAMBDAS <<< "${2:-0 1.0}"

# config.json 이 물체 이름·손·기준 프레임을 갖고 있다. 태스크마다 손으로 적는
# 대신 여기서 읽는다 — 손이 틀리면 전혀 다른 물체의 맵을 쓰게 된다.
eval "$($PY - <<PY
import json
c = json.load(open("$REC/dataset/$TASK/config.json"))
print(f'OBJ={c["object_names"][0]}')
print(f'SIDE={c.get("anchor_hand","right")}')
PY
)"
echo "task=$TASK  object=$OBJ  side=$SIDE  lambdas=${LAMBDAS[*]}"

# 맵을 이미 만들었으면 ROBOTMAP_OVERRIDE 로 지정해 1~3단계를 건너뛴다.
# 예: 수동 물체 맵을 combine_contact_maps.py 로 결합한 경우.
SUFFIX=${SUFFIX:-}
ZSTAR=$REC/heatmap_out/oracle_zstar_${TASK}.npz
HEAT=$REC/heatmap_out/contact_heatmap_${TASK}.npz
MANOMAP=$RET/outputs/mano_robot_map_${SIDE}.npz
ROBOTMAP=${ROBOTMAP_OVERRIDE:-$RET/outputs/robot_contact_map_${TASK}.npz}
RUNDIR=$RET/outputs/sharpa/$SIDE/$TASK/0
LOG=$RET/outputs/exp_logs/$TASK
mkdir -p "$LOG"

if [ -n "$ROBOTMAP_OVERRIDE" ]; then
    echo "ROBOTMAP_OVERRIDE=$ROBOTMAP_OVERRIDE — 맵 생성 단계를 건너뜁니다"
    SKIP_MAPS=1
fi

if [ -z "$SKIP_MAPS" ]; then
# ── 0. MANO -> 로봇 대응 맵 (손 좌우별로 한 번만, 태스크와 무관) ────────────
if [ ! -f "$MANOMAP" ]; then
    echo "── MANO→로봇 대응 맵 생성 ($SIDE)"
    (cd "$RET" && $PY build_mano_robot_map.py --side "$SIDE" \
        --scene "outputs/sharpa/$SIDE/$TASK/0/scene.xml" --out "$MANOMAP") \
        2>&1 | tee "$LOG/00_manomap.log" | tail -4
fi

# ── 1. ray 깊이 보정 오라클 ────────────────────────────────────────────────
echo "── ray 깊이 보정"
(cd "$REC" && $PY oracle_ray_depth.py --raw-dir "dataset/$TASK" --task "$TASK" \
    --object "$OBJ" --side "$SIDE" --report-anchors --out "$ZSTAR") \
    2>&1 | tee "$LOG/01_oracle.log" | tail -8

# ── 2. 접촉 후보 추출 (2 cm, 60도, 손바닥면만) ─────────────────────────────
#    거리·각도 기준은 extract_contact_map.py 의 ANCHOR_MAX_DIST / ANCHOR_CONE_DEG.
echo "── 접촉 맵 추출"
(cd "$REC" && $PY extract_contact_map.py --raw-dir "dataset/$TASK" --task "$TASK" \
    --object "$OBJ" --side "$SIDE" --zstar "$ZSTAR" \
    --palmar-only "$MANOMAP" --sigma-smooth 0.012 --out "$HEAT") \
    2>&1 | tee "$LOG/02_extract.log" | tail -8

# ── 3. 로봇 스킨 접촉 맵 ───────────────────────────────────────────────────
echo "── 로봇 접촉 맵"
(cd "$RET" && $PY build_robot_contact_map.py --contact "$HEAT" --map "$MANOMAP" \
    --out "$ROBOTMAP") 2>&1 | tee "$LOG/03_robotmap.log" | tail -10

fi   # SKIP_MAPS

# ── 4. 조건별 최적화 ───────────────────────────────────────────────────────
BASE=$RET/config/override/do_as_i_do.yaml
cp "$BASE" "$BASE.exp_backup"
trap 'cp "$BASE.exp_backup" "$BASE"; echo "config 복원됨"' EXIT

for L in "${LAMBDAS[@]}"; do
    DST=$RET/outputs/sharpa/$SIDE/${TASK}${SUFFIX}_lam${L}
    echo "── lambda_c = $L"
    # DAID_baseline 과 동일한 가중치에서 시작해, 접촉 항과 warmup 초기화만 더한다.
    # main 에 이미 있는 키는 덮어쓴다 — 뒤에 붙이면 omegaconf 가 중복 키로 거부한다.
    (cd "$RET" && git show main:retargeting/config/override/do_as_i_do.yaml) \
        | grep -vE "^(warmup_min_clearance|warmup_backoff_|ray_depth_path|contact_heatmap_path|robot_contact_|contact_region_)" > "$BASE"
    cat >> "$BASE" <<EOF

# warmup 초기화: 베이스라인의 손바닥 역방향 후퇴는 손이 물체를 감싼 자세에서
# 물체를 관통한다. centroid 기준 역방향은 그 경우에도 멀어진다. 모든 조건 공통.
warmup_min_clearance: 0.2
warmup_backoff_trigger_dist: 0.1
warmup_backoff_mode: centroid

# ray 보정은 접촉 맵의 전제조건이므로 lambda_c = 0 조건에도 켠다.
ray_depth_path: $ZSTAR

# L_contact 는 매핑점->목표 거리의 가중평균(m)이라 점 개수·가중치 합에 무관하다.
contact_heatmap_path: $HEAT
robot_contact_map_path: $ROBOTMAP
robot_contact_paired: true
robot_contact_max_points: 64
robot_contact_alpha: 0.05
robot_contact_skip_warmup: true
robot_contact_repel_scale: 0.0
robot_contact_attract_scale: $L
EOF
    # 태스크별 추가 설정. 예: EXTRA_CFG="force_pedestal_start: false"
    [ -n "$EXTRA_CFG" ] && printf '%s\n' "$EXTRA_CFG" >> "$BASE"
    (cd "$RET" && $CONDA run --no-capture-output -n retargeting python launch.py \
        --task "$TASK" --raw-dir "../reconstruction/dataset/$TASK" \
        --force --no-wait-on-finish) > "$LOG/run${SUFFIX}_lam${L}.log" 2>&1 || true

    # 완료 확인. 예전에 타임아웃으로 죽은 실행의 이전 결과물을 새 결과로 착각한
    # 적이 있어, 로그에 최종 줄이 있을 때만 보관한다.
    if ! tr '\r' '\n' < "$LOG/run${SUFFIX}_lam${L}.log" | grep -q "Final object tracking error"; then
        echo "  [!] 완료되지 않음 — $LOG/run_lam${L}.log 확인"
        continue
    fi
    rm -rf "$DST"; mkdir -p "$DST"; cp -r "$RUNDIR" "$DST/"
    tr '\r' '\n' < "$LOG/run${SUFFIX}_lam${L}.log" | grep "Final object tracking error" | tail -1
done

# ── 5. 지표 ────────────────────────────────────────────────────────────────
echo
echo "═══════════════ $TASK 지표 ═══════════════"
for L in "${LAMBDAS[@]}"; do
    D=$RET/outputs/sharpa/$SIDE/${TASK}${SUFFIX}_lam${L}/0
    [ -f "$D/trajectory_mjwp.npz" ] || continue
    (cd "$RET" && $PY contact_metrics.py --run "$D" --side "$SIDE" \
        --heatmap "$HEAT" --map "$ROBOTMAP" --label "$TASK  lambda_c = $L")
    (cd "$RET" && $PY analyze_grasp_contacts.py --run "$D" --side "$SIDE" \
        --map "$ROBOTMAP" 2>/dev/null | grep -A8 "PALMAR") || true
done
