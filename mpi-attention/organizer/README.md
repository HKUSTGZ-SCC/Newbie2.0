# organizer/ — mpi-attention 组织者工具(不随参赛包分发)

## 工具

| 工具 | 用途 |
|---|---|
| `gen_reference.py` | 生成私有种子 manifest + fp64 参考(必须在组织者纯净仓库上构建 gen_ref) |
| `evaluate_submission.py` | 裁判器:clone → 冻结校验 → 源码扫描 → 3 次运行外部计时 → 独立重算误差 → judge_result.json |

## 私有评测生成(每届比赛一次)

```bash
python3 gen_reference.py \
    --source-manifest /path/to/mpi-attention/manifest.json \
    --ref-dir /judge/reference \
    --out-manifest /judge/manifest.json
# 打印的私有种子务必保密;参考目录不得分发
```

私有 manifest 与公开版**同形状、不同种子**——形状特化是合法优化,
换形状会误伤选手;换种子已足以击穿输出回放。

## 裁判

```bash
# 准备模式(只 clone + 冻结哈希 + 源码扫描,不执行)
python3 evaluate_submission.py --repo <git-url> --revision <sha> \
    --manifest /judge/manifest.json --ref-dir /judge/reference \
    --work-dir /judge/work/team_a --log-dir /judge/logs/team_a

# 完整评测(3 次运行取中位)
python3 evaluate_submission.py ... --execute --runs 3 --timeout-s 1800
```

输出 `<work-dir>/judge_result.json`:`status=accepted|rejected|prepared`、
`score.tiers[]`(逐档中位时间/吞吐)、`score.geo_mean_gflops`(排名主指标)、
`score.speedup_vs_baseline`、`validation.*`(冻结/扫描/一致性闸)、`review_flags`(人工复核清单)。

## 防线与残余风险(必读)

| 防线 | 机制 |
|---|---|
| 改冻结文件 | frozen_files sha256 逐一比对 → rejected |
| 外部数学库 | 源码正则扫描(cblas/mkl/fftw/eigen/dlopen/LD_PRELOAD/链接 flag)→ rejected |
| 伪造输出 | 裁判用 numpy 独立重算 Frobenius 误差,不信任任何自报字段 |
| 削减计算量 | 输入为稠密均匀随机、FLOPs 按固定公式、全量输出校验 |
| 伪造计时 | plausibility 上限(40 TF/s,高于 bf16 物理峰值)+ 外部墙钟一致性闸(overhead_budget_s) |
| 环境注入 | judge 注入 PATH,清空 LD_LIBRARY_PATH/LD_PRELOAD |

**残余风险(必须执行 top-N 人工复核)**:选手脚本理论上可在冻结 bench
落盘后覆写计时 JSON 再跑 report——输出校验强制其至少真实计算一次,
plausibility 封死物理不可能成绩,但"适度虚报时间"无法被完全自动排除。
对前三名:人工 diff `scripts/run_benchmark.sh`、`Makefile`、kernel 源码,
并可用 `--proc N` 固定进程数仲裁复测(脚本 CLI 契约要求支持)。

## 环境备忘(裁判机)

- OpenMPI 4.1.5:`yum install openmpi openmpi-devel`,工具在
  `/usr/lib64/openmpi/bin`(本机无 environment-modules,裁判器已注入 PATH)
- 本机 32+ 网卡导致 OMPI 默认启动扫描 ~70s:入口脚本已固定
  `--mca btl vader,self --mca pml ob1`,启动 ~14s(96 rank,PMIx wireup 为
  O(N²),不可避免);计分只用内核区 MPI_Wtime,不受影响
- 单机 2× Xeon 8476C(96 物理核/192 线程);评测期间应独占机器,
  预期计时方差 1-5%

## 误差阈值标定方法(如需重标定)

三精度变体 × 3 seeds × 3 tiers,统计 max Frobenius 相对误差:
fp32 全程 / bf16 点积+fp32 累加 / bf16 全程(RTNE 转换)。
`tol = max(3 × E_bf16full, 2 × E_fp32)`。
2026-09-08 实测(种子 20260908/1/42):fp32=1.62e-6,bf16dot=7.19e-4,
bf16full=2.43e-3 → **tol=0.01**(bf16-full 约 4 倍裕量)。
标定器源码方法见 competition/README.md 附录说明(不随包分发时以本
文档为准)。

## 参考再生(公开侧自检)

```bash
bash scripts/regen_reference.sh   # 按公开 manifest 确定性再生并核对哈希
```
