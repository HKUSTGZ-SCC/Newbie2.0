#!/usr/bin/env bash
# selftest.sh — mpi-attention 端到端自测(make test 调用)
# ============================================================
# 用小尺寸档位在 ~2 分钟内验证:
#   1. 干净构建零警告
#   2. fp64 参考生成确定性(两次哈希一致)
#   3. MPI baseline 双路径(np=1 / np=3)对参考全 pass
#   3b. 单核朴素串行(无 MPI/OpenMP)对参考全 pass
#   4. 并行度硬校验(np×threads>192 被拒,退出码 2)
#   5. 错误输出被自检捕获(错种子 → FAIL + 退出码 1)
#   6. report 聚合 results.json 且 all_pass 为真
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v mpirun >/dev/null 2>&1; then
    if [[ -x /usr/lib64/openmpi/bin/mpirun ]]; then
        export PATH="/usr/lib64/openmpi/bin:$PATH"
    else
        echo "FAIL: mpirun not found" >&2; exit 3
    fi
fi

MPFLAGS="--mca btl vader,self --mca pml ob1"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PASS=0; FAIL=0
ok()   { echo "  [ok]   $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "== 1. clean build =="
make -C "$ROOT" clean >/dev/null
if make -C "$ROOT" all >/dev/null 2>"$WORK/build_err.log"; then
    if [[ -s "$WORK/build_err.log" ]]; then
        bad "build has stderr output:"; cat "$WORK/build_err.log"
    else
        ok "build clean, zero warnings"
    fi
else
    bad "build failed"; cat "$WORK/build_err.log"; exit 1
fi

echo "== 2. reference determinism =="
"$ROOT/bin/gen_ref" --seed 20260908 --tier tA:4,8,96,32 --tier tB:1,2,200,16 \
    --out-dir "$WORK/ref" >"$WORK/hash1.txt"
"$ROOT/bin/gen_ref" --seed 20260908 --tier tA:4,8,96,32 --tier tB:1,2,200,16 \
    --out-dir "$WORK/ref2" >"$WORK/hash2.txt"
if diff -q "$WORK/hash1.txt" "$WORK/hash2.txt" >/dev/null; then
    ok "gen_ref bit-identical across runs"
else
    bad "gen_ref hashes differ"
fi

echo "== 3. baseline correctness (np=1 and np=3) =="
for NP in 1 3; do
    OMP_NUM_THREADS=1 mpirun -np $NP $MPFLAGS "$ROOT/bin/bench_mpi" \
        --seed 20260908 --tier tA:4,8,96,32 --tier tB:1,2,200,16 \
        --out-dir "$WORK/out$NP" --ref-dir "$WORK/ref" --runs 2 \
        >"$WORK/bench$NP.log" 2>&1 && RC=0 || RC=$?
    if [[ $RC -eq 0 ]] && grep -q "tA.*pass\]" "$WORK/bench$NP.log" \
                      && grep -q "tB.*pass\]" "$WORK/bench$NP.log"; then
        ok "np=$NP both tiers pass"
    else
        bad "np=$NP (rc=$RC)"; cat "$WORK/bench$NP.log"
    fi
done

echo "== 3b. serial (single-core, no MPI/OpenMP) correctness =="
"$ROOT/bin/bench_serial" --seed 20260908 --tier tA:4,8,96,32 \
    --tier tB:1,2,200,16 --out-dir "$WORK/outserial" \
    --ref-dir "$WORK/ref" --runs 1 >"$WORK/serial.log" 2>&1 && RC=0 || RC=$?
if [[ $RC -eq 0 ]] && grep -q "tA.*pass\]" "$WORK/serial.log" \
                  && grep -q "tB.*pass\]" "$WORK/serial.log"; then
    ok "serial single-core both tiers pass"
else
    bad "serial (rc=$RC)"; cat "$WORK/serial.log"
fi

echo "== 3c. 16-core milestone (full load + idle-rank path) =="
OMP_NUM_THREADS=1 mpirun -np 16 $MPFLAGS "$ROOT/bin/bench_mpi" \
    --seed 20260908 --tier tA:4,8,96,32 --tier tB:1,2,200,16 \
    --out-dir "$WORK/out16" --ref-dir "$WORK/ref" --runs 1 \
    >"$WORK/bench16.log" 2>&1 && RC=0 || RC=$?
if [[ $RC -eq 0 ]] && grep -q "tA.*pass\]" "$WORK/bench16.log" \
                  && grep -q "tB.*pass\]" "$WORK/bench16.log"; then
    ok "np=16: tA(32 units, full load) + tB(2 units, 14 idle ranks) pass"
else
    bad "np=16 (rc=$RC)"; cat "$WORK/bench16.log"
fi

echo "== 3d. fp32 variants bit-identical (serial/omp/mpi) =="
"$ROOT/bin/bench_serial" --seed 20260908 --tier tA:4,8,96,32 \
    --out-dir "$WORK/bit_serial" --runs 1 --warmup 0 >/dev/null
OMP_NUM_THREADS=4 "$ROOT/bin/bench_omp" --seed 20260908 --tier tA:4,8,96,32 \
    --out-dir "$WORK/bit_omp" --runs 1 --warmup 0 >/dev/null
OMP_NUM_THREADS=1 mpirun -np 3 $MPFLAGS "$ROOT/bin/bench_mpi" \
    --seed 20260908 --tier tA:4,8,96,32 --out-dir "$WORK/bit_mpi" \
    --runs 1 --warmup 0 >/dev/null
HS=$(sha256sum "$WORK/bit_serial/serial_tA_output.bin" | cut -d' ' -f1)
HO=$(sha256sum "$WORK/bit_omp/omp_tA_output.bin" | cut -d' ' -f1)
HM=$(sha256sum "$WORK/bit_mpi/tA/output.bin" | cut -d' ' -f1)
if [[ "$HS" == "$HO" && "$HS" == "$HM" ]]; then
    ok "serial/omp/mpi outputs bit-identical (${HS:0:12}…)"
else
    bad "bit-identity broken: serial=$HS omp=$HO mpi=$HM"
fi

echo "== 4. parallelism limit enforcement =="
OMP_NUM_THREADS=64 mpirun -np 8 $MPFLAGS "$ROOT/bin/bench_mpi" \
    --seed 20260908 --tier tA:4,8,96,32 --out-dir "$WORK/outbad" \
    >"$WORK/limit.log" 2>&1 && RC=0 || RC=$?
if [[ $RC -eq 2 ]] && grep -q "parallelism limit violated" "$WORK/limit.log"; then
    ok "np×threads>192 rejected with exit 2"
else
    bad "limit not enforced (rc=$RC)"; cat "$WORK/limit.log"
fi

echo "== 5. wrong output must be caught (wrong seed vs reference) =="
OMP_NUM_THREADS=1 mpirun -np 3 $MPFLAGS "$ROOT/bin/bench_mpi" \
    --seed 111 --tier tA:4,8,96,32 --tier tB:1,2,200,16 \
    --out-dir "$WORK/outwrong" --ref-dir "$WORK/ref" --runs 2 \
    >"$WORK/wrong.log" 2>&1 && RC=0 || RC=$?
if [[ $RC -eq 1 ]] && grep -q "FAIL\]" "$WORK/wrong.log"; then
    ok "wrong-seed output flagged FAIL, exit 1"
else
    bad "wrongness not caught (rc=$RC)"; cat "$WORK/wrong.log"
fi

echo "== 6. report aggregation =="
"$ROOT/bin/report" --out-dir "$WORK/out3" --bench mpi-attention \
    --tier tA --tier tB >/dev/null
if python3 - "$WORK/out3/results.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["all_pass"] is True, "all_pass false"
assert len(r["tiers"]) == 2, "tier count"
assert r["geo_mean_gflops"] > 0, "geo mean"
g = 1.0
for t in r["tiers"]:
    g *= t["throughput_gflops"]
g **= 0.5
assert abs(g - r["geo_mean_gflops"]) < 0.02, "geo mismatch"  # 2 位小数舍入容差
print("results.json schema/geo ok")
PYEOF
then ok "results.json valid, geo-mean consistent"; else bad "results.json invalid"; fi

echo
echo "selftest: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
