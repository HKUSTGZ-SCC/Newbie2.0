#!/usr/bin/env bash
# regen_reference.sh — 按公开 manifest 确定性再生 fp64 参考输出
# (参考不入版本库;sha256 冻结在 manifest.json,再生后自动核对)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="${1:-$ROOT/manifest.json}"
OUT_DIR="${2:-$ROOT/data/reference}"

make -C "$ROOT" ref >/dev/null

readarray -t LINES < <(python3 - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["seed"])
for t in m["tiers"]:
    print(f"{t['tier_id']}:{t['B']},{t['H']},{t['S']},{t['D']}")
PYEOF
)
SEED="${LINES[0]}"
ARGS=()
for line in "${LINES[@]:1}"; do
    ARGS+=(--tier "$line")
done

mkdir -p "$OUT_DIR"
"$ROOT/bin/gen_ref" --seed "$SEED" "${ARGS[@]}" --out-dir "$OUT_DIR" | tee /tmp/regen_hashes.txt

# 核对与 manifest 冻结的哈希一致
python3 - "$MANIFEST" /tmp/regen_hashes.txt <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
lines = [l.split() for l in open(sys.argv[2]) if l.strip()]
got = {name: h for name, h, *_ in lines}
ok = True
for t in m["tiers"]:
    if got.get(t["tier_id"]) != t["reference_sha256"]:
        print(f"HASH MISMATCH for {t['tier_id']}: "
              f"manifest={t['reference_sha256'][:16]}… got={got.get(t['tier_id'],'?')[:16]}…")
        ok = False
print("reference hashes: " + ("ALL OK" if ok else "MISMATCH"))
sys.exit(0 if ok else 1)
PYEOF