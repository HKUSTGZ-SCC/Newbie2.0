# reference/ — attention 实现全集(先读本文)

同一定理的四种写法 + 选手契约,全部在本目录:

```
O = softmax(Q·Kᵀ/√D) · V      Q,K,V ∈ [B][H][S][D] 行主序 fp32
```

## 可改性边界(本目录唯一规则)

| 文件 | 归属 |
|---|---|
| `attention_baseline.c` | **选手起点,可改**(计分实现由它长出来) |
| 其余全部(`attention.h`/`attention_variants.h`/`attention_serial.c`/`attention_omp.c`/`attention_ref.c`/`README.md`) | **冻结只读**(sha256 见 `manifest.json` 的 `frozen_files`) |

选手可新增任意源文件并在 Makefile 的 `KERNEL_SRC` 注册;内部代码组织完全自定。

## 差异矩阵(四个实现,每个只差一个轴)

全部为教科书式**两遍 softmax**(分数 → 行 max → exp/Σ → P·V → 归一化)、
纯标量循环、无向量化指示、不物化 S×S(scratch 至多一行 O(S))。

| | serial | omp | ref | **baseline** |
|---|---|---|---|---|
| 精度 | fp32 | fp32 | **fp64** | fp32(选手可降) |
| 并行 | 无 | **OpenMP 线程** | 无(串行) | **MPI 进程** |
| 切分 | — | (b,h) 单元维 | — | (b,h) 单元维轮转 |
| 通信 | — | — | — | 计算零通信 + 一次 Gatherv |
| 角色 | 朴素地板 + 调试工具 | 共享内存参照(不计分) | **真值生成器** | **选手起点(计分)** |
| 实测 geo GF/s | 4.15(1 核) | 35.87(16 线程) | ~3.4(一次性生成) | **39.54**(np=16) |

**位级一致**:serial / omp / baseline 循环逐行同构、同累加顺序 ——
同输入输出位级相同(selftest 常驻检查);ref 与它们仅差 fp32 舍入
(实测 E ≈ 0.0002% ≪ tol = 1%)。

### 用 diff 读差异

```bash
diff -u attention_serial.c attention_omp.c      # +线程
diff -u attention_serial.c attention_ref.c      # +fp64 精度
diff -u attention_omp.c    attention_baseline.c # 线程 → MPI 进程
```

## 选手契约(唯一冻结接口,`attention.h`)

```c
AttStatus attention_forward_mpi(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                MPI_Comm comm,
                                float *restrict O);
```

- 集合调用;每个 rank 拿到**全量** Q/K/V(位级一致,harness 保证)
- 返回 ATT_OK 后 **rank 0 的 O 必须是完整 [B][H][S][D] fp32 结果**
  (内部 gather/归并计入你的耗时)

## 自由度地图(只要结果在 tol 内,以下全部合法)

| 位置 | 可改内容 |
|---|---|
| `attention_baseline.c` + 你新增的任何文件 | 切分维度、通信策略、算法、精度、代码组织(整文件重写也行) |
| `Makefile` | `KERNEL_SRC` 追加源文件;CFLAGS 任意(`-O3 -ffast-math -funroll-loops` 合法) |
| `scripts/run_benchmark.sh` 内部 | np、`OMP_NUM_THREADS`、mpirun 绑定/MCA、分多次 mpirun(逐档不同 np)——CLI 参数语义不能变 |

**硬约束**:① 输出与 fp64 参考 Frobenius 相对误差 ≤ **tol=0.01(即 1%)**;
② `manifest.frozen_files` 不得修改;③ 禁外部数学库(libc/libm、MPI、
OpenMP 运行时除外);④ 并行上限 np ≤ 96、np×threads ≤ 192(harness 硬校验)。

## baseline 三节导读(起点,改什么动哪里)

| 节 | 内容 | 选手主要改动点 |
|---|---|---|
| 1. 切分维度 | B×H 单元轮转(u ≡ rank mod np) | **主战场**:Q 行切分(免通信)/ K·V 行片切分(需 online softmax 可结合状态 + 跨进程归并)/ 2D 混合 |
| 2. 计算 | 两遍 softmax(与 serial 逐行同构) | SIMD(AVX-512/BF16)、分块复用 K/V、降精度、hybrid 线程 |
| 3. 汇总 | Gatherv 到 rank 0(契约) | 换切分维度时同步改(按行 gather、归并散布等) |

已知弱点(= 优化空间):long 档只有 4 个单元,np>4 即大量 rank 闲置
(实测见 manifest notes);每 Q 行整扫 K/V;无 SIMD/精度/重叠优化。

## 取舍:两遍 softmax 与序列维并行为何是选手挑战

两遍 softmax 的 max/Σ 是**全局量**(需先见整行全部分数),因此:

- **单元维切分**(四实现的做法):每个 (b,h) 单元独立完整,两遍天然可行,
  零通信 —— 但 long 档只有 B×H=4 个单元,np>4 时大量 rank 闲置
- **序列维切分**是 long 档头号优化方向:
  - Q 行切分:各 rank 持全量 K/V 只算自己的 Q 行片,免通信
  - K/V 行片切分:max/Σ 跨 rank 归并前无法 exp —— 需要 **online softmax**
    的可结合状态 (m, l, acc)(两遍做不到),或等价两阶段归并方案
    —— 这正是 flash-attention 式的核心思想

## 谁在用这些实现

| 二进制 | 用到的实现 | 说明 |
|---|---|---|
| `bin/bench_mpi`(计分) | baseline(选手可重写) | 冻结 harness 计时 |
| `bin/bench_omp` | omp | 不计分,信息参照(16 线程) |
| `bin/bench_serial` | serial | 不计分,调试/地板 |
| `bin/gen_ref` | ref | 生成 fp64 参考(公开 + 私有) |

**下一步阅读**:[../../competition/README.md](../../competition/README.md)(赛制、
评分、合法/非法清单)、[../../manifest.json](../../manifest.json)(档位/tol/冻结清单)。
