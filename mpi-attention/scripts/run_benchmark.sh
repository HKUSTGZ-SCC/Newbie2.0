#!/usr/bin/env bash
# run_benchmark.sh — mpi-attention 唯一评测入口
# ============================================================
# CLI 契约(冻结,裁判器按此调用;脚本内部实现选手可改):
#   bash scripts/run_benchmark.sh --manifest <manifest.json> \
#        --ref-dir <fp64参考目录> --out-dir <输出目录> \
#        [--runs <R>] [--proc auto|N]
#
# 行为:
#   1. 确保 OpenMPI 工具链可用(必要时注入 PATH)
#   2. make -j 全量构建
#   3. 单次 mpirun 依次跑 manifest 中的全部档位(摊销 mpirun 启动开销)
#   4. bin/report 聚合 <out-dir>/results.json
#
# 选手可改:NP(每档 rank 数策略)、mpirun 绑定/MCA 参数、
#           是否分多次 mpirun、编译参数(见 competition/README.md)。
# 选手不可改:CLI 参数语义、结果文件格式、计时与正确性判定(在冻结二进制内)。
set -euo pipefail

# ---------- 仓库根定位 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 默认参数(选手可调) ----------
NP_DEFAULT=16                       # --proc auto 时使用的 rank 数(本版默认 16 核)
MPRUN_BIND="--bind-to core"         # 绑定策略(合法调优面)
MPRUN_MCA="--mca btl vader,self --mca pml ob1"  # 单机共享内存通信,避开网卡扫描
WARMUP=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"  # 默认 1 线程/rank;选手可调
                                                  # (约束:np×threads ≤ 192)

# ---------- CLI 解析 ----------
MANIFEST="" REF_DIR="" OUT_DIR="" RUNS=5 PROC=auto
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --ref-dir)  REF_DIR="$2";  shift 2 ;;
        --out-dir)  OUT_DIR="$2";  shift 2 ;;
        --runs)     RUNS="$2";     shift 2 ;;
        --proc)     PROC="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MANIFEST" && -n "$REF_DIR" && -n "$OUT_DIR" ]] || { echo "usage: $0 --manifest M --ref-dir R --out-dir O [--runs N] [--proc auto|N]" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

# ---------- 工具链 ----------
if ! command -v mpicc >/dev/null 2>&1; then
    if [[ -x /usr/lib64/openmpi/bin/mpicc ]]; then
        export PATH="/usr/lib64/openmpi/bin:$PATH"
    else
        echo "mpicc not found; install openmpi-devel or set PATH" >&2; exit 3
    fi
fi

# ---------- 构建 ----------
make -C "$ROOT" -j"$(nproc)" all >/dev/null

# ---------- 读取 manifest(python3 仅作解析,不参与计时) ----------
readarray -t TIER_LINES < <(python3 - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{m['seed']}")
print(f"{m['tolerance']['tol']}")
for t in m["tiers"]:
    print(f"{t['tier_id']}:{t['B']},{t['H']},{t['S']},{t['D']}")
PYEOF
)
SEED="${TIER_LINES[0]}"
TOL="${TIER_LINES[1]}"
TIER_ARGS=()
TIER_IDS=()
for line in "${TIER_LINES[@]:2}"; do
    TIER_ARGS+=(--tier "$line")
    TIER_IDS+=("${line%%:*}")
done

# ---------- 参考输出 ----------
REF_ARGS=()
if [[ -d "$REF_DIR" ]]; then
    REF_ARGS=(--ref-dir "$REF_DIR")
fi

# ---------- rank 数 ----------
NP="$NP_DEFAULT"
if [[ "$PROC" != "auto" ]]; then NP="$PROC"; fi

mkdir -p "$OUT_DIR"

# ---------- 运行(单次 mpirun 依次跑全部档位) ----------
set +e
mpirun -np "$NP" $MPRUN_BIND $MPRUN_MCA \
    "$ROOT/bin/bench_mpi" \
    --seed "$SEED" --runs "$RUNS" --warmup "$WARMUP" --tol "$TOL" \
    "${TIER_ARGS[@]}" "${REF_ARGS[@]}" \
    --out-dir "$OUT_DIR"
BENCH_RC=$?
set -e

# ---------- 聚合结果(即使有档位未过自检也生成 results.json) ----------
REPORT_ARGS=()
for id in "${TIER_IDS[@]}"; do
    REPORT_ARGS+=(--tier "$id")
done
"$ROOT/bin/report" --out-dir "$OUT_DIR" --bench mpi-attention "${REPORT_ARGS[@]}"

echo "results written to $OUT_DIR/results.json"
exit $BENCH_RC
