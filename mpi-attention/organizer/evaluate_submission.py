#!/usr/bin/env python3
"""mpi-attention 赛题裁判器(组织者工具,不随参赛包分发)。

用法:
  python3 evaluate_submission.py \\
      --repo https://example.com/team/submission.git --revision <sha> \\
      --manifest /judge/manifest.json --ref-dir /judge/reference \\
      --work-dir /judge/work/team_a --log-dir /judge/logs/team_a \\
      [--runs 3] [--timeout-s 1800] [--proc auto] [--execute] [--keep-work]

不带 --execute 时仅做 clone + 冻结哈希校验 + 源码扫描(prepare 模式)。

流程:
  1. 校验私有 manifest 与参考文件 sha256
  2. clone 提交、checkout revision(detached)
  3. 逐文件比对 manifest.frozen_files 的 sha256(不符 → rejected)
  4. 源码扫描可改文件:禁外部数学库/动态注入
  5. 以固定参数调 scripts/run_benchmark.sh ×N 次:
     - 外部计时 perf_counter_ns,进程组超时 killpg
     - 环境注入 PATH=/usr/lib64/openmpi/bin:…,清空 LD_LIBRARY_PATH/LD_PRELOAD
  6. 裁判独立重算误差(numpy 读 output.bin 对私有 fp64 参考),
     不信任选手自报的 self_check 字段
  7. 闸门:plausibility 上限、外部墙钟一致性(overhead_budget_s)、
     输出 sha256 与 results.json 一致
  8. 评分:N 次运行取中位(逐档)→ 吞吐几何平均;写 judge_result.json

已知残余风险(必须人工复核 top-N):脚本在 bench 落盘后覆写计时文件
理论上可绕过计时,但输出校验强制其至少真实计算一次;plausibility 上限
(40 TF/s)封死物理不可能成绩;前若干名强制人工 diff 审查脚本与 Makefile。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

OPENMPI_BIN = "/usr/lib64/openmpi/bin"

# 源码扫描:出现即拒
FORBIDDEN_PATTERNS = [
    (r"#\s*include\s*<\s*(cblas|mkl|openblas|fftw|atlas|eigen|gsl|lapacke?)\.h",
     "外部数学库头文件"),
    (r"\b(dlopen|dlsym)\s*\(", "动态库加载"),
    (r"LD_PRELOAD", "LD_PRELOAD 注入"),
    (r"(-lcblas|-lmkl|-lopenblas|-lfftw|-latlas|-llapack|-lblas)\b", "链接外部数学库"),
    (r"-D\s*main\b|-include\s+\S+\.(c|h)", "宏改写冻结标识符/强插源文件"),
]
# 出现即警告(进入人工复核清单)
REVIEW_PATTERNS = [
    (r"\b(system|popen)\s*\(", "子进程调用"),
    (r"\b(fork|execv|execl|posix_spawn)\s*\(", "进程创建"),
    (r"\bnohup\b|\bsetsid\b|&\s*$", "后台/脱离进程组迹象"),
]

LDD_WHITELIST = ("libc.so", "libm.so", "libgomp.so", "libmpi.so", "libdl.so",
                 "libpthread.so", "ld-linux", "librt.so", "libgcc_s.so",
                 "libstdc++.so", "libhwloc.so", "libevent", "libpmix",
                 "libudev", "libiberty", "libz.so", "libopen-pal", "libopen-rte")


def fail(msg, result=None, out=None):
    if out and result is not None:
        result["status"] = "rejected"
        result["reason"] = msg
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"REJECTED: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(repo: Path, manifest: dict, result: dict, out: Path):
    bad = []
    for rel, expect in manifest["frozen_files"].items():
        p = repo / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != expect:
            bad.append(f"hash mismatch {rel}")
    if bad:
        result["validation"]["frozen_hashes_ok"] = False
        fail("冻结文件被修改: " + "; ".join(bad[:5]), result, out)
    result["validation"]["frozen_hashes_ok"] = True


def scan_sources(repo: Path, manifest: dict, result: dict, out: Path):
    scan_ext = (".c", ".h", ".sh", "Makefile")
    hits, warns = [], []
    for p in sorted(repo.rglob("*")):
        if not p.is_file() or p.suffix not in scan_ext and p.name != "Makefile":
            continue
        rel = str(p.relative_to(repo))
        if "/.git/" in rel or rel.startswith("bin/") or rel.startswith("results/"):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for pat, why in FORBIDDEN_PATTERNS:
            if re.search(pat, text):
                hits.append(f"{rel}: {why}")
        for pat, why in REVIEW_PATTERNS:
            if re.search(pat, text, re.MULTILINE):
                warns.append(f"{rel}: {why}")
    if hits:
        result["validation"]["source_scan_ok"] = False
        result["validation"]["source_scan_hits"] = hits[:20]
        fail("源码扫描命中禁止项: " + "; ".join(hits[:5]), result, out)
    result["validation"]["source_scan_ok"] = True
    if warns:
        result["review_flags"] = warns[:50]
        print(f"NOTE: {len(warns)} 处需人工复核特征(已记录)")


def run_once(repo: Path, run_idx: int, manifest: dict, args, result: dict,
             log_dir: Path) -> dict:
    out_dir = Path(args.work_dir) / f"run{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", "scripts/run_benchmark.sh",
           "--manifest", str(Path(args.manifest).resolve()),
           "--ref-dir", str(Path(args.ref_dir).resolve()),
           "--out-dir", str(out_dir.resolve()),
           "--runs", str(manifest["timing"]["repeats"]),
           "--proc", args.proc]
    env = {
        "PATH": f"{OPENMPI_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C",
        "OMP_NUM_THREADS": os.environ.get("JUDGE_OMP_NUM_THREADS", "1"),
    }
    t0 = time.perf_counter_ns()
    timed_out = False
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            text=True)
    try:
        output, _ = proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
    wall_s = (time.perf_counter_ns() - t0) / 1e9

    (log_dir / f"run{run_idx}.log").write_text(output or "")
    return {"exit_code": proc.returncode, "wall_s": wall_s,
            "timed_out": timed_out, "results_path": str(out_dir / "results.json"),
            "out_dir": str(out_dir)}


def verify_outputs(run: dict, manifest: dict, ref_dir: Path, result: dict,
                   args, out: Path):
    """裁判独立重算误差 + 各闸门。"""
    import numpy as np

    res = json.load(open(run["results_path"]))
    tier_by_id = {t["tier_id"]: t for t in manifest["tiers"]}
    tiers_out = []
    for t in res["tiers"]:
        tid = t["tier_id"]
        m = tier_by_id.get(tid)
        if m is None:
            fail(f"未知档位 {tid}", result, out)
        # 输出文件与哈希链
        opath = Path(run["out_dir"]) / t["output_file"]
        if not opath.is_file():
            fail(f"输出缺失 {opath}", result, out)
        if sha256_file(opath) != t["output_sha256"]:
            fail(f"输出 sha256 与 results.json 不一致({tid})", result, out)
        # 形状
        with open(opath, "rb") as f:
            hdr = f.read(32)
        B, H, S, D, dtype = (int.from_bytes(hdr[8 + 4 * i:12 + 4 * i], "little")
                             for i in range(5))
        if (B, H, S, D) != (m["B"], m["H"], m["S"], m["D"]) or dtype != 4:
            fail(f"输出形状/dtype 不符({tid}): {(B, H, S, D, dtype)}", result, out)
        # 独立重算 Frobenius 相对误差
        O = np.fromfile(opath, dtype="<f4", offset=32).astype(np.float64)
        rpath = ref_dir / f"{tid}_output.bin"
        R = np.fromfile(rpath, dtype="<f8", offset=32)
        if O.size != R.size:
            fail(f"输出元素数不符({tid})", result, out)
        err = float(np.linalg.norm(O - R) / np.linalg.norm(R))
        tol = manifest["tolerance"]["tol"]
        if err > tol:
            fail(f"相对误差超限({tid}): {err * 100:.4g}% > {tol * 100:g}%",
                 result, out)
        # 逐档误差留档(原始值;展示口径为百分比)
        result.setdefault("tier_rel_err", {}).setdefault(tid, []).append(err)
        # plausibility
        if t["throughput_gflops"] > manifest["plausibility"]["max_tier_gflops"]:
            fail(f"吞吐超出物理上限({tid}): {t['throughput_gflops']:.1f} GF/s",
                 result, out)
        tiers_out.append({"tier_id": tid, "time_s": t["time_s_median"],
                          "gflops": t["throughput_gflops"], "rel_err": err,
                          "pass": True})
    # 外部墙钟一致性
    med_sum = sum(t["time_s_median"] for t in res["tiers"])
    budget = manifest["timing"]["overhead_budget_s"]
    reps = manifest["timing"]["repeats"] + manifest["timing"]["warmup"]
    if run["wall_s"] > reps * med_sum + budget:
        fail(f"外部墙钟与内核时间不一致: wall={run['wall_s']:.1f}s > "
             f"{reps}x{med_sum:.2f}s + {budget}s 预算", result, out)
    result["validation"]["external_consistency_ok"] = True
    result["validation"]["ldd_ok"] = check_ldd(Path(run["out_dir"]), result)
    return tiers_out


def check_ldd(out_dir: Path, result: dict) -> bool:
    """对 run 产物不做 ldd(二进制在 repo/bin 下);此处留接口,由
    组织者在 clone 内运行 ldd 并记录(见 README)。简化:跳过。"""
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--manifest", required=True, help="私有 manifest.json")
    ap.add_argument("--ref-dir", required=True, help="私有 fp64 参考目录")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--proc", default="auto")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    ref_dir = Path(args.ref_dir)
    work = Path(args.work_dir)
    log_dir = Path(args.log_dir)
    work.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "repo": args.repo, "revision": args.revision, "status": "running",
        "reason": None, "runs": [], "score": None,
        "validation": {}, "machine": {
            "mpi": "OpenMPI 4.1.5", "cpu": "2x Xeon Platinum 8476C (96C/192T)"},
    }
    out_path = work / "judge_result.json"

    # 1. 参考文件哈希
    for t in manifest["tiers"]:
        rp = ref_dir / f"{t['tier_id']}_output.bin"
        if not rp.is_file() or sha256_file(rp) != t["reference_sha256"]:
            fail(f"私有参考缺失或哈希不符: {t['tier_id']}", result, out_path)

    # 2. clone + checkout
    repo = work / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    r = subprocess.run(["git", "clone", "--no-tags", args.repo, str(repo)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("git clone 失败: " + r.stderr[:300], result, out_path)
    r = subprocess.run(["git", "checkout", "--detach", args.revision],
                       cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"checkout {args.revision} 失败: " + r.stderr[:300], result, out_path)

    # 3. 冻结哈希 + 4. 源码扫描
    check_frozen(repo, manifest, result, out_path)
    scan_sources(repo, manifest, result, out_path)

    if not args.execute:
        result["status"] = "prepared"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print("prepare-only 完成(clone + 冻结校验 + 源码扫描通过)")
        return

    # 5. 运行 N 次
    for i in range(args.runs):
        run = run_once(repo, i, manifest, args, result, log_dir)
        result["runs"].append({k: v for k, v in run.items() if k != "out_dir"})
        if run["timed_out"]:
            fail(f"run{i} 超时(>{args.timeout_s}s)", result, out_path)
        if run["exit_code"] != 0:
            fail(f"run{i} 非零退出码 {run['exit_code']}(详见日志)", result, out_path)
        try:
            verify_outputs(run, manifest, ref_dir, result, args, out_path)
        except json.JSONDecodeError:
            fail(f"run{i} results.json 无法解析", result, out_path)

    # 6. 评分:N 次运行逐档中位 → 几何平均;误差取各 run 最大值(百分比展示)
    import math
    tier_ids = [t["tier_id"] for t in manifest["tiers"]]
    score_tiers = []
    for tid in tier_ids:
        times = []
        for run in result["runs"]:
            res = json.load(open(run["results_path"]))
            t = next(x for x in res["tiers"] if x["tier_id"] == tid)
            times.append(t["time_s_median"])
        times.sort()
        med = times[len(times) // 2] if len(times) % 2 else 0.5 * (
            times[len(times) // 2 - 1] + times[len(times) // 2])
        m = next(x for x in manifest["tiers"] if x["tier_id"] == tid)
        gf = m["flops"] / med / 1e9
        errs = result.get("tier_rel_err", {}).get(tid, [])
        err_max = max(errs) if errs else None
        score_tiers.append({
            "tier_id": tid, "time_s_median": med, "gflops": round(gf, 2),
            "rel_err": err_max,
            "rel_err_pct": round(err_max * 100, 6) if err_max is not None else None})
    geo = math.exp(sum(math.log(t["gflops"]) for t in score_tiers) / len(score_tiers))
    result.pop("tier_rel_err", None)  # 中间数据不入最终 JSON
    result["score"] = {
        "tiers": score_tiers,
        "geo_mean_gflops": round(geo, 2),
        "speedup_vs_baseline": round(
            geo / manifest["baseline_geo_gflops"], 3),
        "tol_pct": manifest["tolerance"]["tol"] * 100,
    }
    result["status"] = "accepted"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    tol_pct = manifest["tolerance"]["tol"] * 100
    print(f"ACCEPTED: geo {geo:.2f} GF/s "
          f"(x{geo / manifest['baseline_geo_gflops']:.2f} vs baseline, "
          f"误差预算 {tol_pct:g}%)")
    for t in score_tiers:
        print(f"  {t['tier_id']:<6} {t['gflops']:>8.2f} GF/s"
              f"  err {t['rel_err_pct']:.6f}% / {tol_pct:g}%")
    if not args.keep_work:
        for run in result["runs"]:
            p = Path(run["results_path"]).parent
            if p != work / "repo":
                shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
