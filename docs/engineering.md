# 工程化行为与验证边界

## 目标与结构

采用小型 Python 包、TOML 和普通子进程，不引入服务、任务队列、容器集群或复杂配置框架。
`watermarking/` 的生成、攻击、统计距离与 Cython 算法代码保持原样。
`2-textgen.py`、`3-detect.py` 成为兼容入口，实际逻辑分别放在 `cpd/generation.py` 和 `cpd/detection.py`。
原始 R 与 Python SeedBS 脚本保留；`cpd/seedbs.py` 提取原 Python 数值函数，`cpd/segmentation.py` 负责文件读取、坐标与 NOT 选择。

先前已通过的小模型 CSV 保存于 `tests/fixtures/generation_baseline.json`。
本机原始工作区补丁和入口备份还保留在被 Git 忽略的 `results/pre-engineering-baseline/`。
没有擅自创建 Git 提交；后续提交时应把 `cpd/`、`configs/`、`tests/`、文档和依赖文件一起提交。

## 明确改变的行为

| 原行为 | 新行为 |
|---|---|
| 本地 Dataset 任意加载错误都回退在线 C4 | 默认报本地错误；`dataset_source=online` 才明确读取在线 C4，不是回退 |
| 加载模型后才发现没有可用 prompt | 先加载 tokenizer 和本地数据，扫描至目标数量或 `max_scan_examples`，不足则停止 |
| 输出同名前缀直接覆盖 | 单独生成入口拒绝已存在结果；统一运行器每次创建新目录并加锁 |
| Mac 自动设备选择难以测试 | 显式 `device=cpu/cuda/auto`；配置与 Bash 小测试默认 CPU |
| 参数散落且未知字段可能被忽略 | 单次实验 TOML；未知字段和错误类型直接拒绝，实际默认值完整存档 |
| 一个 seed 被 squeeze 成标量 | 保留 `reshape(-1).tolist()` 修复；seed CSV 始终为 1×N |
| 检测 genfromtxt 导致单样本变一维 | 整数 CSV 显式保留二维，检查 seed 与样本数量和词表范围 |
| 检测越界索引被错误截断 | 零基索引越界明确报错；默认第 0 个样本；窗口 -1 表示该样本全部窗口 |
| 未知模型尝试下载后直接 raise | 统一入口从本地 config.json 读取词表；旧检测入口要求 `--vocab_size` |
| 检测参数 seed 没有控制全局置换 RNG | 每个样本/窗口显式设置稳定任务 seed，使独立进程和串行运行一致 |
| 运行结果缺少代码和输入身份 | 保存实际配置、包版本、Git 状态、源文件副本、输入 SHA256、日志、阶段状态及结果 SHA256 |

`--save` 仍表示完整输出前缀；只有省略它时才使用 `output_dir/smoke`，帮助文字已改正。
直接调用数字脚本不自动创建完整实验记录；需要可追踪的运行请使用 `python -m cpd run`。

## 保留的科学计算语义

- `tokens_count=64, buffer_tokens=20` 实际采样 84 个新 token；prompt 默认 50，因此模型返回总长 134。
- `truncate_vocab=8` 仍在采样后裁剪高位 token ID；没有改成采样前屏蔽。
- 默认替换 `[0:0]`、插入长度 0 保留，仍调用原函数并消耗随机数。
- 即使零攻击，仍 decode / encode；长序列取 `[1:65]`，不足 64 时左补 0。未重定义 special token 行为。
- 检测保留原两种统计量的顺序（编辑距离、非编辑距离）和边界 NaN。
- 偶数窗口参数 W 的外层切片有 W+1 个 tokens，内部统计比较长度 W 的子窗口；保留原实现，未悄悄纠正定义。
- transform 统计量保留原脚本对 `null=False` 的处理，未替换检验算法。

如果换 tokenizer，`truncate_vocab`、首 token 丢弃、补零的意义仍需科学核对。
统一检测/分段入口当前要求插入长度为零，因为原生成后处理可能得到不同长度的行。
非零插入可使用独立生成入口，但不属于已验证的统一流程契约。
统一配置不开放回译或 GPT 改写；原生成 CLI 仍保留其参数，但未在本轮验证。

## SeedBS 与 NOT

正式配置默认 `backend="r"`：`scripts/segment_original.R` 从原 `4-seedbs.R` 读取并执行四个数值函数，从 `5-not.R` 提取并执行原 NOT 循环，适配只处理配置、文件和输出记录。R block size 固定为原来的 10；每个 interval 按原独立脚本方式重置配置 seed，两个 metric 按原 `apply` 顺序执行。R 入口在本机未执行过；缺少 Rscript 时 preflight 明确停止，不自动切换到 Python。

下面描述的是仅在显式 `backend="python"` 时使用的工程测试适配。它不作为原 R 算法已等价复现的证据。

SeedBS 沿用 `4.1-seedbs.py` 的区间、KS 和 block permutation 数值函数。
置换次数和 block 大小成为参数；为每个 sample/metric/interval 设置稳定 seed，顺序不再隐式决定随机状态。
这改变了原批量脚本的随机数流，不能与历史结果按同一 seed 直接宣称一致。

NOT 选择按 `5-not.R` 原逻辑：过滤显著区间，选择最短区间的候选点，删除覆盖该点的其他区间，重复并排序。
并列最短取输入中的第一个，与 R 的 `which.min` 规则一致。
本次没有搬运论文绘图、真值列表、假阳性统计或 Rand index 评估；这些仍在原 R 分析脚本中。

坐标约定：

- `detect/<sample>-<position>.csv` 的 sample、position 从 0 开始。
- 分段 CSV 的 sample、metric、interval 从 0 开始；metric=0 为编辑距离，metric=1 为非编辑距离。
- `from`、`to` 沿用原 SeedBS 的 1 基、闭区间坐标；读取 pvalue 行时换算为 `[from-1:to]`。
- `index_within_segment` 是原 KS split 位置；`change_point_index = from + index_within_segment - 1`，沿用原 R 公式。
- `segment_length = to - from` 沿用原排序量，并不是闭区间行数（后者多 1）。

没有 Rscript 的本机只测试 Python 数值函数与原 Python 的一致性、NOT 构造例，以及整个小流程可复跑。
**R/Python 等价、真实模型水印质量、检验功效和论文数值结果不在“工程测试通过”的含义内。**

## 运行记录与限制

每阶段独立进程，生成结束后释放模型内存。R 阶段额外记录 `r-session.txt`，包含实际 R 环境版本。日志同时显示在终端并写入阶段 `run.log`。
子进程失败保存完整 traceback 和退出码，父进程停止后续阶段。`run.json` 的 `ready` 表示阶段成功、等待后续；`succeeded` 表示分段或全流程完成。
正常 Ctrl-C 会终止子进程组；kill -9、断电等无法保证记录结束状态，需要人工核对锁和残留进程。

接续阶段核对配置、源代码、Python 环境、输入文件以及上阶段结果哈希，拒绝混用不同实验。
不支持阶段内部断点续跑，也不支持覆盖重试；当前优先保证小规模运行易于审计。
输入权重和数据没有复制，运行记录保存原路径及内容哈希。迁移时需要同时迁移或重新准备这些输入；`source/` 也不包含可跨系统使用的编译产物。
在线数据只记录配置和可见本地文件，不能保证流式源在将来不变；正式可复现实验应使用本地快照。

CPU 小模型测试证明同一环境的确定性。不同硬件、CUDA 算子、依赖版本或 dtype 仍可能引起数值差异。
`requirements-macos-py312.lock` 只记录已验证 Mac 环境，不伪装为已经验证的 Linux/CUDA 锁文件。

## 后续验收顺序

1. 在 AutoDL 按 README 安装并重新编译扩展，运行离线小样例。
2. 准备真实且匹配的模型/tokenizer、本地小 Dataset，先执行 `--stage generate`。
3. 从同一运行目录接续检测、分段；记录时间、内存与输出契约。
4. 在 R 环境对同一 pvalue 输入核对 SeedBS/NOT，评估科学实现差异。
5. 达到这些验证后再决定并行或性能优化，避免先引入高维护基础设施。
