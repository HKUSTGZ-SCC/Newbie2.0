# mpi-attention

校内赛第三题:**纯 CPU、纯 C、OpenMPI** 的 scaled dot-product attention
吞吐赛。无任何外部数学库(GEMM/softmax/通信全手写),fp64 参考验证 +
相对误差阈值,比拼三档负载的 GFLOP/s 几何平均。

**选手从这里开始**:读 [competition/README.md](competition/README.md)(题面)、
[competition/SETUP.md](competition/SETUP.md)(环境)、
[competition/SUBMISSION.md](competition/SUBMISSION.md)(提交契约)。

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH   # OpenMPI(yum install openmpi openmpi-devel)
bash scripts/regen_reference.sh             # 再生 fp64 参考(~1 分钟)
make run                                    # 编译 + 跑全部档位
```

- 单核朴素串行(无 MPI/OpenMP,不计分):**4.15 GF/s** geo
- OpenMP 参照(16 线程,不计分):35.87 GF/s geo
- 基线(MPI baseline,np=16):**39.54 GF/s** geo(本版默认 16 核,上限 96;
  全部实现为两遍 softmax、纯标量、单元维切分,long 档刻意留白——
  序列维并行是选手头号优化空间)
- 理论天花板:fp32 ≈ 16 TF/s,bf16 ≈ 32 TF/s(受误差阈值约束)

全部 attention 实现(三参照 + 选手起点 baseline + 契约)集中在
`src/reference/`,差异对照 + 自由度地图一表可辨:
[src/reference/README.md](src/reference/README.md)。

组织者工具在 [organizer/](organizer/)(私有参考生成 + 裁判器),不随参赛包分发。
评测协议与冻结哈希见 [manifest.json](manifest.json)。
