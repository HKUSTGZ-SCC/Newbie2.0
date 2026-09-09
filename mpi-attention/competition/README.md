# mpi-attention:纯 CPU MPI 注意力吞吐赛

在 **2× Xeon Platinum 8476C(96 物理核,AVX-512/BF16)** 上,用 **纯 C +
OpenMPI**(OpenMP 可混合)实现 scaled dot-product attention,比拼三档工作负载
的吞吐几何平均。**禁止一切外部库**(BLAS/MKL/FFTW/Eigen 等;仅 libc/libm、
MPI、OpenMP 运行时),GEMM、softmax、通信全部手写。

```
O = softmax(Q·Kᵀ/√D) · V        Q,K,V ∈ [B][H][S][D] fp32
```

## 三档工作负载

| 档 | B | H | S | D | FLOPs | 考察点 |
|---|---|---|---|---|---|---|
| short | 48 | 16 | 512 | 64 | 51.5 G | (b,h) 单元切分、SIMD、缓存/pack、纯算力 |
| mixed | 4 | 8 | 2048 | 128 | 68.7 G | 切分选择 + 分块 |
| long | 1 | 4 | 8192 | 64 | 68.7 G | 序列维切分、online softmax 跨进程归并、通信计算重叠 |

**评分** = 三档吞吐(GFLOP/s,按 `4·B·H·S²·D` 折算)的**几何平均**。
计时口径:每档 warmup 1 次 + 5 次计时(每次取最慢 rank),取中位;
裁判 3 次运行再取中位。

## 基准与参照(2026-09-08 实测,本版默认 16 核)

| 实现 | short | mixed | long | geo |
|---|---|---|---|---|
| 单核朴素串行(无 MPI/OpenMP,不计分) | 4.19 | 4.10 | 4.15 | 4.15 GF/s |
| OpenMP 参照(单进程 16 线程,不计分) | 54.6 | 58.6 | 14.4 | 35.87 GF/s |
| **MPI baseline**(你的起点,np=16) | 60.2 | 63.1 | 16.3 | **39.54 GF/s** |

参照阶梯(本版默认 16 核):单核朴素(4.15 GF/s)→ OpenMP(35.87)
→ MPI baseline(39.54,9.5×)。**注意 long 档**:B×H=4 个单元,单元维
切分最多用满 4 核(np=96 抽查仅 5.9 GF/s,比 np=16 更慢)——序列维并行
是头号优化空间(见下)。单核朴素版(`make serial`)同时是**调试工具**。

理论天花板参考:fp32 AVX-512 FMA ≈ 16 TF/s,bf16 点积 ≈ 32 TF/s
(注意误差阈值约束)。优化空间约 **三个数量级**。

## 快速开始

```bash
bash scripts/regen_reference.sh        # 再生 fp64 参考(一次性,~1 分钟)
make run                               # 编译 + 跑全部档位(默认 np=16)
cat results/results.json               # 逐档时间/吞吐/误差自检
```

环境搭建见 [SETUP.md](SETUP.md);提交规则见 [SUBMISSION.md](SUBMISSION.md)。

## 正确性判据

- **输入种子分公开/私有两层**:公开 manifest 的种子(20260908)仅供本地
  开发自测;正式评测使用组织者**私有种子**(独立生成,不随参赛包分发,
  哈希冻结在私有 manifest),评测时由裁判注入你的入口脚本——你无法预知
  评测输入,输出也无法离线准备
- 你的输出与 **fp64 参考实现**的 Frobenius 相对误差 ≤ **0.01(即 1%)**:
  `‖O−R‖_F / ‖R‖_F ≤ tol`(误差展示口径为百分比)
- **计算精度完全自由**:fp32 恒过(实测 E≈0.0002%);bf16 点积+fp32 累加
  舒适通过(实测 E≈0.07%);bf16 全程约 4 倍裕量(实测 E≈0.24%)。
  更激进的混合精度自担风险

## 接口契约(冻结,签名不得改)

```c
/* src/reference/attention.h(冻结) */
AttStatus attention_forward_mpi(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                MPI_Comm comm,
                                float *restrict O);
```

- 集合调用;每个 rank 拿到**全量** Q/K/V(位级一致)
- 返回 ATT_OK 后 **rank 0 的 O 必须是完整 [B][H][S][D] fp32 结果**
  (内部 gather/归并计入你的耗时)
- 你的工作面:`src/reference/attention_baseline.c`(起点,可整文件重写)
  + 新增任意文件(Makefile 注册);其余冻结文件见 manifest

## 可自由优化(合法面)

- **序列维并行(long 档头号得分点)**:Q 行切分(免通信,各 rank 持全量
  K/V 只算自己的 Q 行片)、K/V 行片切分 + 跨进程归并(两遍 softmax 的
  max/Σ 是全局量,需 online softmax 可结合状态 (m,l,acc) 或等价方案,
  见 src/reference/README.md"取舍")、2D 混合、Iallreduce 计算通信重叠、
  单轮自定义 MPI_Op……
- 微核:SIMD(AVX-512 F/BF16/VNNI)、寄存器分块、pack、K/V 跨 Q 行复用
- 任意精度:fp32/bf16/fp16/混合(误差阈值内)
- rank × OpenMP 线程布局(约束:**np ≤ 96,np×threads ≤ 192**,harness 硬校验)、
  mpirun 绑定/MCA 参数、NUMA 亲和
- 编译参数(CFLAGS 可改:`-O3 -ffast-math -funroll-loops …` 均合法)
- 跨调用 scratch 缓存(warmup 吸收一次性成本)

## 非法操作

- 一切外部数学库(BLAS/MKL/OpenBLAS/ATLAS/FFTW/Eigen/GSL/LAPACK…)
- 修改冻结文件(manifest.json 的 `frozen_files` 清单,sha256 校验)
- 修改输入 Q/K/V、伪造/回放输出、伪造计时
- 识别评测 case 或缓存其结果(评测种子私有,与公开开发种子不同)
- 后台进程、联网、注入 LD_PRELOAD

## 目录导览

```
src/reference/  ← attention 实现全集(先读其 README:差异对照 + 自由度地图)
                  ├ attention_baseline.c  你的起点(唯一预置可改文件)
                  ├ attention.h          冻结接口契约
                  └ serial/omp/ref       冻结参照(地板/参照条/真值)
src/harness/    ← 冻结计时/报告(bench/report/gen_ref 入口)
src/common/     ← 冻结 PRNG/数据/IO/SHA-256/误差
scripts/        ← run_benchmark.sh(唯一入口,内部可改)/ selftest / regen_reference
manifest.json   ← 档位/种子/tol/冻结哈希
```

## baseline 策略说明(起点)

- **单元轮转 + 两遍 softmax**(刻意朴素,单一路径):rank r 整算单元
  u ≡ r (mod np),各 rank 零通信独立计算,最后 Gatherv 汇总到 rank 0
- 与 serial/omp 同一种算法、同循环同顺序(三者输出位级一致)
- 内部按**切分 / 计算 / 汇总**三节组织,改什么动哪里见其文件头导读
- **已知弱点 = 你的优化空间**:long 档只有 4 个单元(np>4 即闲置 rank);
  每 Q 行整扫 K/V(内存带宽受限);无 SIMD/分块/精度优化/通信重叠
