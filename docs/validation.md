# 本机工程验证记录

## 首次 AutoDL 前准备复验（2026-09-12）

本轮在干净的 `main` / `daf3f09` 基线上工作，未 commit/push。未发现仓库或父目录 AGENTS.md 文件。
最新 `python -m unittest discover -s tests -v`：**13 项通过，57.158 秒，退出码 0**，仍仅代表 macOS arm64 / Python 3.12.4 / CPU 验证。

| 本轮项目 | 结果 |
|---|---|
| Bash 完整入口，gumbel，N=1/3，batch=1，50/64/20 | 两次正常退出；CSV 均存在 |
| seeds / prompt / before / attacked 形状 | N=1：1×1 / 1×50 / 1×84 / 1×64；N=3：1×3 / 3×50 / 3×84 / 3×64 |
| CSV 内容 | Mac 历史基线一致；所有平台均执行整数范围、prompt 和后处理内容校验 |
| 人工数据准备脚本 | 临时 tokenizer 下离线保存/读回 3 条 text、长度检查与文本哈希通过；拒绝覆盖 |
| 下载准备安全边界 | 非完整 revision 被拒绝；真正的模型下载分支未执行 |
| generate-only check | 无 R 可通过 CPU 检查；不调用模型权重加载；CUDA 不可用明确返回 2 |
| 通用依赖安装预演 | offline dry-run 配合已安装 Torch 精确约束退出 0，Would make no changes |
| 环境和构建 | 64 个已安装包的依赖检查通过；两个本机 Cython 扩展导入通过 |
| Git 范围 | 未发现已跟踪的 .venv、权重、数据缓存、build/.so/.pyc；新增忽略规则不是替代跟踪检查 |
| 代码检查 | 编译检查、Bash 语法和 git diff --check 通过；原算法指纹测试通过 |

本轮没有未解决的本地测试失败；非法输入、重复输出、缺 R/CUDA 的非零退出属于预期拒绝测试。
模型下载命令仅新增并检查帮助及前置拒绝逻辑，没有用真实 OPT 权重验证。
**服务器待验证：Linux 依赖解析/安装与 Cython 编译、CUDA 驱动/架构和 dtype/显存、网络与许可、OPT 模型/tokenizer 实际行为、真实模型生成、原 R 流程。**
操作顺序与官方资料链接见 [AutoDL 首次运行手册](autodl-first-run.md)。

## 工程重构时的历史验证

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
