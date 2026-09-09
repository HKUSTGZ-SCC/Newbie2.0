# 非评测工具与归档

本目录不属于五样例评测的运行路径。

- `cache_builder/`：组织者在更新原始样例后，重新生成预处理输入和 RNG 缓存时使用。
- `upstream_nonbenchmark/`：保留的上游训练、部署服务、基础模型推理和通用交互脚本，仅供溯源。

这些归档脚本的相对路径已随归档变化，不能作为可执行入口。比赛运行请只使用
`scripts/run_compute_benchmark.sh`。
