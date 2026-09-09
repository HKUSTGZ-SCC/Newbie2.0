# SETUP — 环境搭建

已验证平台:TencentOS Server 4,gcc 12.3.1,OpenMPI 4.1.5(yum 源自带)。

## 1. 安装 OpenMPI

```bash
sudo yum install -y openmpi openmpi-devel
```

**PATH 坑**:本机没有 environment-modules,`module load` 不可用。
mpicc/mpirun 装在 `/usr/lib64/openmpi/bin`,必须手动导出:

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
mpicc --version    # 应显示 gcc (Tencent Compiler 12.3.1.8) 12.3.1
```

建议写进 `~/.bashrc`。`run_benchmark.sh` 找不到 mpicc 时会自动注入该路径。

## 2. 构建

```bash
make all          # bench_mpi / bench_omp / gen_ref / report
make test         # 端到端自测(约 1 分钟,6 项检查)
```

OpenMPI 在非默认路径时:`make CC=/usr/lib64/openmpi/bin/mpicc`。

## 3. 再生 fp64 参考(首次必做)

参考输出不入版本库(可确定性再生,sha256 冻结在 manifest.json):

```bash
bash scripts/regen_reference.sh     # ~1 分钟,末尾应显示 ALL OK
```

## 4. 跑分

```bash
make run                              # 默认 np=16(本版默认 16 核), OMP_NUM_THREADS=1
# 或自定义:
bash scripts/run_benchmark.sh --manifest manifest.json \
     --ref-dir data/reference --out-dir results --runs 5
```

## 已知坑

1. **mpirun 启动慢(本机特性)**:机器有 32+ 网卡,OMPI 默认组件扫描
   ~70s 且警告刷屏。入口脚本已固定
   `--mca btl vader,self --mca pml ob1`(单机共享内存),启动约 14s。
   计分只用内核区 MPI_Wtime,不受启动影响。
2. **以 root 跑**:OpenMPI 默认拒绝 root。本机普通用户无需处理;
   若必须 root,加 `--allow-run-as-root`(或环境变量)。
3. **OpenMP 混合**:默认 `OMP_NUM_THREADS=1`。改 rank×线程布局时记住
   harness 硬校验 `np×threads ≤ 192`,越界直接退出码 2。
   混合并行建议配合 `mpirun --bind-to socket`(给每 rank 留线程空间)
   并设 `OMP_PROC_BIND=TRUE`。
4. **NUMA**:双路 48 核/socket。`--bind-to core` 默认即可;
   观察 `numactl -H` 了解节点分布。
5. **-march=native**:评测与编译同一台机器,安全;CFLAGS 可自由改
   (-O3/-ffast-math/-funroll-loops 合法)。
6. **计时噪声**:多租户时段方差可达 5%;以中位数口径 + 独占时段为准。

## OpenMP 参照(不计分,信息参照)

```bash
make omp
OMP_NUM_THREADS=16 bin/bench_omp --seed 20260908 \
    --tier short:48,16,512,64 --tier mixed:4,8,2048,128 --tier long:1,4,8192,64 \
    --out-dir /tmp/ompbar --ref-dir data/reference
```

展示"单进程共享内存"的合理水平(geo ≈ 35.9 GF/s @ 16 线程)。
与 serial/baseline 同为两遍 softmax、单元维切分、纯标量循环。

## 单核朴素串行(不计分,调试/最底层参照)

```bash
make serial
# 小档快速验证(推荐先跑这个确认数据管线):
bin/bench_serial --seed 20260908 --tier tA:4,8,96,32 \
    --out-dir /tmp/s && ls /tmp/s
# 真实档位(--runs 1,每遍几十秒):
bin/bench_serial --seed 20260908 --tier short:48,16,512,64 \
    --out-dir /tmp/s --ref-dir data/reference --runs 1
```

无 MPI、无 OpenMP(链接里没有 libmpi/libgomp),教科书式两遍 softmax,
单核约 3-4 GF/s——**真实档位每遍几十秒,只建议 --runs 1 或小档调试用**。
正确性自检与其它 harness 相同(`--ref-dir` + `[pass]`)。
