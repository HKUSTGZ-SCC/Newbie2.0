# UnifoLM-WMA 固定计算基准

这个目录是从 `ASC26-Embodied-World-Model-Optimization` 固定抽取的五个
`case1` 输入快照。它只包含评测所需的输入和参考视频，不依赖比赛输入
目录的绝对路径。

## 已固定的输入

| 数据集 | videoid | frame stride | 源样例的交互轮数 |
| --- | ---: | ---: | ---: |
| `unitree_z1_stackbox` | 5 | 4 | 12 |
| `unitree_z1_dual_arm_stackbox` | 5 | 4 | 7 |
| `unitree_z1_dual_arm_stackbox_v2` | 5 | 4 | 11 |
| `unitree_z1_dual_arm_cleanup_pencils` | 0 | 4 | 8 |
| `unitree_g1_pack_camera` | 0 | 6 | 11 |

`world_model_interaction_prompts/` 是原始输入快照（PNG、H5、归一化统计量和
一行 CSV）；`ground_truth/` 是对应的参考视频。`manifest.json` 是机器可读的
固定配置，评测程序应以它为准。

## 预处理输入与 RNG 快照

发布的评测包已经包含这些缓存，参赛者不需要也不应重新生成。组织者仅在更新原始
样例时，才需要在**相同的软件栈、相同的 `CUDA_VISIBLE_DEVICES` 拓扑**下执行：

```
bash tools/cache_builder/prepare_compute_benchmark_inputs.sh
```

它在 `preprocessed_inputs/` 为每个 case 写一个缓存。每个缓存包含原始
`prepare_init_input` 的归一化/空间变换结果、状态和动作维度、提示词、帧率，
以及在“模型、数据和输入队列初始化完成，首次 DDIM 采样之前”捕获的 Python、
NumPy、PyTorch CPU 和**所有可见 CUDA 设备**的 RNG 状态。

随后直接运行 `bash scripts/run_compute_benchmark.sh`；该脚本默认要求这些缓存
存在，命中后不读取 CSV/PNG/H5，也不构造 `WMAData`。缓存带有配置 SHA-256、
分辨率、视频长度、frame stride 和观测步数校验；配置变化或 CUDA 可见设备数量
变化会拒绝使用缓存，必须重新生成。缓存是由组织者在可信环境生成的 `torch.save` 文件
（含 RNG 对象），随评测包交付；不可从不可信来源加载。

这里没有缓存 `image_guided_synthesis_sim_mode` 内的图像编码、文本编码、VAE、
DDIM 或任何后续轮次的 observation：它们属于要考核的模型计算；第 2 轮及以后
的输入也依赖前一轮生成结果，预存会改变完整交互任务。

## 环境与模型

使用 Python 3.10.18，并按仓库 README 安装项目。缓存命中的评测路径不导入
`h5py` 或 `external/dlimp`；两者只在组织者重新制作输入缓存时需要。
GPU 版 PyTorch/torchvision 必须与本机 CUDA 驱动和 `xformers==0.0.27` 兼容；
项目锁定的 API 版本为 `torch==2.3.1`、`torchvision==0.18.1`。

仓库附带的 `scripts/create_compute_benchmark_env.sh` 会创建隔离的 Python
3.10.18 环境，并安装评测所需 GPU 运行时和项目依赖。

下载的官方 Dual checkpoint 应位于：

```
downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
```

它的 Hugging Face LFS SHA-256 是：

```
f5c5202b41bbda777e2a861104a8886e9c2b0cd981d496a8bffaa67e0149c7ef
```

## 固定推理配置

- seed `123`，并设置 `PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；
  启动代码还会固定 Python、NumPy、PyTorch CPU/CUDA RNG，关闭 cuDNN benchmark
  和 TF32，并启用 deterministic algorithms。预处理缓存会在首次采样前恢复同一
  RNG 检查点，避免跳过数据路径后改变随机流。
- batch size `1`，输入分辨率 `320×512`，视频长度 `16`，动作维度和状态维度 `16`。
- DDIM `50` steps、`eta=1.0`、`uniform_trailing`、`guidance_rescale=0.7`、
  `perframe_ae=true`、guidance scale `1.0`。
- 默认使用每个样例在源启动脚本中的完整轨迹：`12/7/11/8/11` 轮。评测时间应覆盖
  该样例完整视频生成过程；不要为了计时把 `n_iter` 截断为 1。仅本地 smoke test
  时才可显式设置 `BENCHMARK_N_ITER=1`。

从仓库根目录运行：

```
bash scripts/run_compute_benchmark.sh
```

仅用于本地单轮调试：

```
SEED=123 BENCHMARK_N_ITER=1 bash scripts/run_compute_benchmark.sh
```

结果默认写到 `outputs/compute_benchmark/`，该目录不会进入版本控制。
每个 case 的阶段计时会写到 `outputs/compute_benchmark/profiles/<dataset>.json`。
运行结束时终端还会输出 `Pure model compute time`：这是所有 decision 和
world-model generation 的 CUDA elapsed time，不含任何加载、预处理和保存；如需
关闭该输出，传入 `--no-print_compute_time`。

## 计时口径

不要把进程导入、checkpoint 读取/反序列化、模型搬运到 GPU、CSV/H5/PNG 读取、
CUDA 上下文初始化、TensorBoard/MP4 写入混入“中间模型计算”时间。建议分别报告：

1. `cold_start`：进程启动至第一个可计时推理前；
2. `warmup`：一次不计分的固定输入推理（随后重新设 seed）；
3. `model_compute`：完整 `n_iter` 中所有 `image_guided_synthesis_sim_mode` 调用的
   GPU elapsed time，前后 `torch.cuda.synchronize()`；
4. `end_to_end`：如有需要，单独报告，不与 `model_compute` 排名混合。

计时 JSON 同时保留 wall-clock 与 CUDA-event 时间。`decision_generation` 与
`simulation_generation` 的 `cuda_s` 是完整轨迹中所有轮次的纯 GPU 时间；
`iteration_result_write`、`full_video_write` 则量化完整视频落盘的代价。
