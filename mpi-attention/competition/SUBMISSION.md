# SUBMISSION — 提交契约

## 提交内容

- 仓库 URL + 不可变 commit SHA(评测以该 SHA 为准)
- `SUBMISSION_NOTES.md`(可选,说明优化思路、rank/线程布局选择)

## 必须保留

| 项 | 契约 |
|---|---|
| `scripts/run_benchmark.sh` | CLI 契约:`--manifest M --ref-dir R --out-dir O [--runs N] [--proc auto\|N]`;裁判以固定参数调用;`--proc N` 时必须全部档用该 rank 数(仲裁用) |
| `src/reference/attention.h` | `attention_forward_mpi` 签名与语义(集合调用,返回后 rank 0 持完整 O) |
| `src/` 下 `manifest.frozen_files` 列出的全部文件 | sha256 逐一校验,改动即拒(`src/reference/attention_baseline.c` 不在冻结清单内,是你的起点) |
| Makefile target 语义 | `make all/bench/omp/serial/ref/report/test/run/clean`(`serial` 为单核朴素参照工具,不计分) |
| 产物格式 | `<out>/<id>/output.bin`(ATTBIN01 头)、`<out>/<id>.json`、`<out>/results.json` 由冻结 `bin/report` 生成 |

## 语义约束

- attention 数学定义、[B][H][S][D] 布局、FLOPs 折算公式(4·B·H·S²·D)不可变
- 输入由冻结 PRNG 按私有种子生成,不得预处理/缓存识别
- 输出与 fp64 参考的 Frobenius 相对误差 ≤ 0.01,即 **1%**(裁判用私有种子独立重算,展示口径为百分比)
- 并行度:1 ≤ np ≤ 96,np × OpenMP 线程 ≤ 192(冻结 harness 硬校验)

## 评分

- 每档:内核区 MPI_Wtime(Allreduce 取最慢 rank),warmup 1 + 5 次取中位
- 裁判 3 次完整运行,逐档再取中位 → GFLOP/s → 三档**几何平均**
- 闸门:plausibility ≤ 40 TF/s;外部墙钟 ≤ (1+5)×Σ中位 + 60s;
  输出 sha256 与 results.json 一致
- 并列展示:vs baseline(39.54 GF/s,np=16)加速比、vs OpenMP 参照(35.87 GF/s,16 线程)

## 作弊边界(举例)

| 行为 | 判定 |
|---|---|
| 改 `src/common/prng.c` 的生成逻辑 | 违规(冻结文件) |
| 链接 `-lopenblas` / `#include <mkl.h>` / `dlopen` | 违规(外部库) |
| Makefile 加 `-Dmain=xxx` 换掉冻结入口 | 违规(改写冻结语义) |
| 输出写死上次结果/识别 seed 切换路径 | 违规(伪造) |
| bf16 点积 + fp32 累加 | **合法**(E≈0.07% ≪ 1%) |
| Q 行切分免通信 | **合法** |
| allgather 后各算各的 | **合法** |
| 跨调用缓存 pack 好的数据结构 | **合法**(warmup 吸收) |
| 改 OMP_NUM_THREADS / mpirun 绑定参数 | **合法** |
| 用识别"评测模式"的分支降低计算量 | 违规(削减语义) |

## 评测环境

见 [SETUP.md](SETUP.md)。评测机器:2× Xeon Platinum 8476C,
OpenMPI 4.1.5,gcc 12.3.1(TencentOS 4)。评测期间独占机器。
