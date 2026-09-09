# 固定五样例计算评测包

本仓库的比赛入口只有一个：

```bash
bash scripts/run_compute_benchmark.sh
```

它以 `examples/compute_benchmark/manifest.json` 中的五个固定 case 为准，完整执行
源样例的 `12/7/11/8/11` 轮交互生成。默认从
`examples/compute_benchmark/preprocessed_inputs/` 读取已经准备好的输入张量和首次采样前
的 RNG 状态；因此不读取 CSV、PNG、H5，也不构造 `WMAData`。终端的
`Pure model compute time` 是所有 decision/world generation 的 CUDA 时间，不含加载和保存。

评测运行时只需关注：

| 路径 | 用途 |
| --- | --- |
| `scripts/run_compute_benchmark.sh` | 五个固定 case 的唯一启动入口 |
| `scripts/evaluation/world_model_interaction.py` | 缓存加载、确定性恢复、完整生成和计时 |
| `scripts/evaluation/eval_utils.py` | 视频 TensorBoard 输出工具 |
| `configs/inference/world_model_interaction.yaml` | 唯一推理配置 |
| `examples/compute_benchmark/` | manifest、缓存、原始输入快照和参考视频 |

checkpoint 固定位置为
`downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt`。详细的输入、RNG、环境和计时口径见
[`examples/compute_benchmark/README.md`](examples/compute_benchmark/README.md)。

`tools/` 中的是组织者工具和上游非评测功能归档，不是比赛运行依赖；参赛者无需阅读或执行。
