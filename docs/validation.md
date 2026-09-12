# 本机工程验证记录

日期：2026-09-12。环境：macOS arm64，Python 3.12.4，Transformers 4.40.2，PyTorch 2.10.0。全程 CPU、离线，无真实模型下载。

`python -m unittest discover -s tests -v`：11 项测试全部通过（42.199 秒）。

- 单样本与三样本必须通过 Bash 入口，CSV 内容与重构前基线一致。
- gumbel 和 transform 检测统计量与原代码在相同 RNG 状态下输出一致。
- 原水印算法源码、原 R / Python SeedBS 脚本的指纹未变化。
- Python SeedBS 数值函数与原 Python 版本对照通过。
- Python 小流程两次独立运行的所有科学结果 CSV 一致。
- 分阶段接续、重复输出拒绝、单样本检测维度、无效索引、缺输入、数据不足、失败日志和未知配置拒绝通过。
- R 不可用时明确停止，不静默回退 Python。

最后按 README 从新样例目录再次运行 `check` 和 `run`，均退出 0。
验证配置（显式 `backend="python"`）：
`results/engineering-final-fixture/smoke.toml`

保留的运行目录：
`results/engineering-final-fixture/runs/tiny-offline-20260912T074057Z-6009ee38/`

生成 CSV 为 1×1、1×50、1×84、1×64；检测 64 个 1×2 CSV；生成、检测、Python 分段阶段退出码均为 0。
模型及输入数据的本地身份、源代码与资源副本、依赖锁文件、日志和结果哈希可在运行目录检查。
这些本地大文件位于 Git 忽略的 results 下；长期可提交的回归证据在 tests/fixtures 下。

未验证：AutoDL/CUDA、真实模型和真实数据、R 适配入口执行、R/Python 统计等价、论文结论与规模性能。
正式配置默认 R，直接加载原 R 数值函数及 NOT 循环；本机没有 Rscript，因此不能声称正式 R 全流程已验收。
工程测试证明的是接口、可追踪性和同环境小样例复跑能力，不等于完成论文科学复现。
