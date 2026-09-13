# Segmenting Watermarked Texts From Language Models

Implementation of the methods described in "Segmenting Watermarked Texts From Language Models" by [Xingchi Li](https://xingchi.li), [Guanxun Li](https://guanxun.li), [Xianyang Zhang](https://zhangxiany-tamu.github.io).

[![NeurIPS Proceedings](https://img.shields.io/badge/NeurIPS%20Proceedings-Segmenting%20Watermarked%20Texts%20From%20Language%20Models-007bff.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a8d295871250443f9747d239925b89d-Abstract-Conference.html)
[![OpenReview](https://img.shields.io/badge/OpenReview-Segmenting%20Watermarked%20Texts%20From%20Language%20Models-8c1b13.svg)](https://openreview.net/forum?id=FAuFpGeLmx)
[![doi](https://img.shields.io/badge/doi-10.48550/arXiv.2410.20670-b31b1b.svg)](https://doi.org/10.48550/arXiv.2410.20670)

首次上服务器请按 [AutoDL 首次运行手册](docs/autodl-first-run.md) 操作；该手册区分本地已验证与服务器待验证，并提供独立 Torch 构建安装及显式输入准备命令。

## 本地工程入口

这份仓库现在提供不依赖 Slurm 的 **生成 → 滚动检测 → SeedBS → NOT 选择** 小流程。
论文的水印采样与检测统计量仍位于 `watermarking/`；原始 R 程序、Slurm 脚本和 rebuttal 材料保留。
正式配置默认直接调用原 R 函数和 NOT 循环；小样例显式选择 Python 验证适配。Python 分段不是已经验证等价的 R 替代品；真实模型、AutoDL/CUDA、R 适配入口和论文统计复现仍需另行验收。

### 1. 安装环境

在仓库根目录执行，建议 Python 3.12（配置解析要求 Python ≥3.11）。正式 R 分段还需要系统安装 R 并使 `Rscript` 位于 PATH；Python 小样例不需要 R：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
python 1-setup.py build_ext --inplace
uv pip check --python .venv/bin/python
```

编译需要 C 编译器及系统开发工具。必须在目标操作系统与 Python 环境内重新编译，不能把 Mac 的 `.so` 复制到 Linux。
`requirements.txt` 是 Mac 便捷入口，引用通用依赖并选择 Torch 2.10.0；`requirements-common.txt` 不选择 Torch 构建，供 Linux 先单独安装 Torch 后配合精确约束使用。`requirements-macos-py312.lock` 只记录 Mac 全部依赖。
Linux/CUDA 的 Torch 平台依赖仍需在目标服务器验证，每次运行也会保存实际版本。

### 2. 先跑完全离线的小样例

以下只构造随机小模型，不下载真实模型，不需要 API key，也不使用 GPU。
`--output` 必须是不存在的新目录；重复准备时换一个名称。

```bash
python scripts/create_smoke_fixture.py --output results/tiny-fixture
python -m cpd check --config results/tiny-fixture/smoke.toml
python -m cpd run --config results/tiny-fixture/smoke.toml
```

程序打印新的运行目录。预期退出码 `0`，其中：

| 位置 | 预期结果 |
|---|---|
| `generate/sample-seeds.csv` | 1×1，无表头 |
| `generate/sample-prompt.csv` | 1×50，无表头 |
| `generate/sample-tokens-before-attack.csv` | 1×84，无表头 |
| `generate/sample-attacked-tokens.csv` | 1×64，无表头 |
| `detect/0-0.csv` 到 `detect/0-63.csv` | 每个 1×2；边界窗口为 NaN |
| `segment/seedbs.csv` | 每个区间、每个统计量的候选位置和显著性，有表头 |
| `segment/changepoints.csv` | NOT 选择后的变点；允许只有表头，表示未选出变点 |
| `run.json` | `status: succeeded`，三个阶段退出码均为 0 |

小样例使用 `segmentation.backend="python"`、水印 key 长度 8、置换次数 3，仅验证程序接口，不代表原 R 实现已经通过验收或有统计意义的实验。
运行回归与失败场景测试：

```bash
python -m unittest discover -s tests -v
```

测试包含必须经过 Bash 启动的单样本和三样本生成，参数为 `50/64/20`、key 长度 1000。
macOS arm64 会与重构前 CSV 完整对比；跨平台要求形状、整数范围、prompt 与后处理内容及流程检查，不声称浮点结果逐位相同。

### 3. 准备真实实验

复制并编辑 `configs/local.toml`，同一次实验只维护这一份配置。
路径相对 **配置文件所在目录** 解析；也可填写绝对路径。

```text
model_root/<model>/model/          # model.save_pretrained，包含 config.json 与权重
model_root/<model>/tokenizer/      # 匹配的 tokenizer.save_pretrained
 dataset_root/allenai/c4/realnewslike/train/  # Dataset.save_to_disk 的具体训练 split
```

数据每条必须有字符串 `text`；默认设置至少需编码为 114 tokens。
在线 C4 必须显式配置 `dataset_source = "online"`。本地加载失败不会触发在线回退。
可复现实验推荐先保存本地 split；在线流式数据不保证未来内容相同。

真实配置默认 `segmentation.backend="r"`，直接加载原 `4-seedbs.R` 数值函数和 `5-not.R` 的 NOT 循环，不执行其中的绘图。全阶段检查会在模型加载前确认 `Rscript` 可用；若暂时只验证生成，可用 `check --stage generate`。

先检查，再只运行生成；AutoDL 使用时显式设置 `device = "cuda"`：

```bash
python -m cpd check --config configs/local.toml --stage generate
python -m cpd run --config configs/local.toml --stage generate
```

确认生成输出后，使用打印出的完整运行目录继续（把 `/path/to/run` 替换为真实路径）：

```bash
python -m cpd run --config configs/local.toml --stage detect --run-dir /path/to/run
python -m cpd run --config configs/local.toml --stage segment --run-dir /path/to/run
```

新实验一次执行全部阶段：

```bash
python -m cpd run --config configs/local.toml --stage all
```

继续运行要求配置、源代码、Python 环境、输入文件及前一阶段 CSV 保持一致。
失败阶段不会自动重跑或覆盖；排错后创建新实验。当前按样本顺序串行执行，真实检测可能很慢。

### 4. 结果、日志与排错

每次新实验自动建立 `名称-UTC时间-随机后缀` 目录：

```text
run/
├── config.toml                 # 原始配置
├── config.resolved.json        # 包含默认值和绝对路径的实际配置
├── environment.json            # Python、系统、所有包版本
├── inputs.json                 # 输入文件路径、大小、时间和 SHA256
├── source.json / source/       # 代码指纹及源文件快照
├── run.json                    # Git 提交/脏工作区状态、时间、退出码、CSV 哈希
├── generate/                   # CSV、run.log、runtime.json
├── detect/                     # CSV、耗时、run.log、runtime.json
└── segment/                    # SeedBS / NOT CSV、run.log、runtime.json
```

默认拒绝同名文件。独立生成 CLI 的 model_root/dataset_root 默认分别为当前目录下的 `models`/`data`，不再使用作者的集群绝对路径。生成前先检查参数、路径、可用文本数量及模型上下文长度；`check` 不加载模型权重，文本可用性在实际生成前检查。
首次运行需要读取输入文件计算 SHA256，大模型/数据会花时间，这是保存输入身份的一部分。
参数错误返回非零退出码；阶段失败会保存完整 traceback，并停止后续阶段。Ctrl-C 会停止子进程并记录中断。
断电或强制 kill 后可能留下 `.running` 锁；核对文件里的 PID 已不存在后才手动移除。

详细行为变化、坐标约定及未验证范围见 [docs/engineering.md](docs/engineering.md)，本机实测记录见 [docs/validation.md](docs/validation.md)。

### 5. 目录职责与旧入口

| 路径 | 职责 |
|---|---|
| `cpd/config.py`、`validation.py` | 配置解析、便宜的运行前检查 |
| `cpd/runtime.py`、`worker.py` | 实验目录、阶段执行、状态与日志 |
| `cpd/generation.py`、`detection.py` | 生成和检测入口，调用原水印函数 |
| `scripts/segment_original.R` | 正式 R 接口，直接读取原算法函数与 NOT 循环 |
| `cpd/seedbs.py`、`segmentation.py` | 显式选择的 Python 小流程适配，未证明与 R 等价 |
| `watermarking/` | 论文算法与两个 Cython 扩展 |
| `configs/`、`tests/` | 实验配置示例、离线回归证据 |
| `2-textgen.py`、`3-detect.py` | 保留原命令名称的兼容入口 |
| `4-seedbs.R`、`5-not.R`、`4.1-seedbs.py` | 原研究脚本，保留作对照 |

`bash scripts/run_textgen_smoke.sh --help` 仍可单独测试生成，默认 CPU；统一运行记录由 `python -m cpd run` 提供。
原 Slurm 批量脚本未作为新的推荐入口维护。原始 `5-not.R` 同时包含绘图、评估和硬编码实验列表，不能直接把新目录传给它；R 适配入口直接执行其原 NOT 循环，不包含论文绘图和 Rand index 评估。

## Citation

```bibtex
@inproceedings{NEURIPS2024_1a8d2958,
  author = {Li, Xingchi and Li, Guanxun and Zhang, Xianyang},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {14634--14665},
  publisher = {Curran Associates, Inc.},
  title = {Segmenting Watermarked Texts From Language Models},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/1a8d295871250443f9747d239925b89d-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```

<details closed>
<summary>OpenReview and ArXiv</summary>

```bibtex
@inproceedings{
  li2024segmenting,
  title={Segmenting Watermarked Texts From Language Models},
  author={Xingchi Li and Guanxun Li and Xianyang Zhang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=FAuFpGeLmx}
}

@misc{li2024segmentingwatermarkedtextslanguage,
  title={Segmenting Watermarked Texts From Language Models}, 
  author={Xingchi Li and Guanxun Li and Xianyang Zhang},
  year={2024},
  eprint={2410.20670},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2410.20670}, 
}
```

</details>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/doccstat/llm-watermark-cpd.svg)](https://starchart.cc/doccstat/llm-watermark-cpd)
