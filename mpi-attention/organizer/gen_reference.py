#!/usr/bin/env python3
"""生成私有评测用 manifest 与 fp64 参考(组织者工具)。

用法:
  python3 gen_reference.py --source-manifest manifest.json \\
      --ref-dir /judge/reference --out-manifest /judge/manifest.json \\
      [--seed <N>]            # 缺省随机生成并打印
      [--repo-dir .]          # 组织者的纯净赛题仓库(用于构建 gen_ref)

产出:
  1. <ref-dir>/<tier>_output.bin  私有种子参考(由纯净仓库的 gen_ref 生成)
  2. <out-manifest>               私有 manifest(同形状、新种子、新哈希)

注意:必须用组织者持有的纯净仓库构建 gen_ref(冻结文件),不得使用
选手提交的任何代码。私有种子不要泄露;参考目录不要分发。
"""
import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--repo-dir", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    src = json.load(open(args.source_manifest))
    seed = args.seed if args.seed is not None else secrets.randbits(48)
    print(f"private seed: {seed} (请妥善保管)")

    repo = Path(args.repo_dir)
    ref_dir = Path(args.ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # 构建纯净 gen_ref(组织者仓库)
    subprocess.run(["make", "-C", str(repo), "ref"], check=True,
                   stdout=subprocess.DEVNULL)

    cmd = [str(repo / "bin" / "gen_ref"), "--seed", str(seed)]
    for t in src["tiers"]:
        cmd += ["--tier", f"{t['tier_id']}:{t['B']},{t['H']},{t['S']},{t['D']}"]
    cmd += ["--out-dir", str(ref_dir)]
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("gen_ref 失败")

    # 回填私有 manifest
    priv = dict(src)
    priv["seed"] = seed
    for t in priv["tiers"]:
        p = ref_dir / f"{t['tier_id']}_output.bin"
        if not p.is_file():
            sys.exit(f"参考未生成: {p}")
        t["reference_output"] = str(p)
        t["reference_sha256"] = sha256_file(p)
        t.pop("baseline_time_s", None)
        t.pop("omp_ref_time_s", None)
    Path(args.out_manifest).write_text(
        json.dumps(priv, indent=2, ensure_ascii=False))
    print(f"private manifest written: {args.out_manifest}")
    print("下一步:evaluate_submission.py --manifest 该文件 --ref-dir 该目录")


if __name__ == "__main__":
    main()
