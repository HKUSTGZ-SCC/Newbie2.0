# ASC 2026：UnifoLM-WMA 固定五样例计算评测

本仓库同时包含公开参赛包源材料和仅供组织者使用的裁判工具。构建可直接分发给参赛队伍的精简包：

```bash
bash scripts/build_competitor_package.sh
```

生成的 `dist/ASC2026-unifolm-wma-competition/` 只含公开开发缓存、推理源码、配置和提交说明；不含 checkpoint、原始样例、训练/部署代码、组织者工具、私有测试输入或参考结果。

公开开发集的本地运行入口是：

```bash
bash scripts/run_compute_benchmark.sh
```

评测必需路径：

- [`README_BENCHMARK.md`](README_BENCHMARK.md)：输入、环境、计时和复现规则。
- `scripts/run_compute_benchmark.sh`：固定 case 的唯一启动脚本。
- `scripts/evaluation/world_model_interaction.py`：运行、缓存恢复和计时实现。
- `configs/inference/world_model_interaction.yaml`：唯一模型配置。
- `examples/compute_benchmark/`：manifest、已准备缓存、输入快照和参考视频。
- `competition/`：将随参赛包分发的 README、提交接口和规则。
- `organizer/evaluate_submission.py`：仅供裁判端使用的 GitHub 提交自动评测器。

训练、部署服务、基础模型推理和通用交互脚本不参与评测，已保存在
`tools/upstream_nonbenchmark/` 仅供溯源；参赛者无需阅读。组织者重新制作缓存时使用
`tools/cache_builder/`。最终私有 case、参考输出和 checkpoint 必须在隔离裁判环境中以只读方式提供，不能放进分发包。
