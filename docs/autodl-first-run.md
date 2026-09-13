# 首次 AutoDL 运行手册

本手册只覆盖：同版本部署 → 离线生成验证 → OPT-1.3B 小规模真实模型生成。
截至本次准备，仅 macOS/CPU 通过验证；Linux、CUDA、下载、真实模型及 R 入口均为服务器待验证。
现有入口是 `python -m cpd check/run --config ... --stage generate`；不使用 Slurm，不创建另一套运行框架。

## 1. 创建实例前，先确认而不是直接购买

确认 GPU 型号/架构、单卡可用显存、主存、Linux 镜像的 Python 与 NVIDIA 驱动版本、C 编译器和 Python 开发头文件；不要只看镜像名称中的 CUDA 数字。
确认可写磁盘位置、可用容量、系统盘/数据盘持久化及备份规则。本文不假定 `/mnt/data` 等路径存在。
确认计费模式、单价、余额、存储收费及关机后的数据保留期。断开 SSH 不代表关机；付费数据盘可能在关机后继续计费。以控制台和 [AutoDL 计费说明](https://www.autodl.com/docs/price/) 为准。
本轮不创建或启动实例、不连接服务器，也不生成实例管理脚本。

候选选择 `facebook/opt-1.3b`：它是仓库原来支持的模型，保留原 `truncate_vocab=8` 配置，参数规模小于原 Llama-3-8B 候选；不为了首次部署切换算法或模型家族。

## 2. SSH 配置好后，确认终端确实在远程

下面第一条在 Mac 终端执行；地址、端口和用户名必须来自用户自己的实例连接信息，不把密码或密钥写进脚本：

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
```

连接后在服务器执行（主机名和系统信息应与实例吻合）：

```bash
hostname
uname -srm
pwd
command -v python3.12
nvidia-smi
```

随后设置用户确认过的可写路径；后续命令都在这个远程终端执行。以下是占位符，必须替换：

```bash
CPD_WORKSPACE='/absolute/writable/workspace'
CPD_STORAGE='/absolute/writable/persistent/storage'
mkdir -p "$CPD_WORKSPACE" "$CPD_STORAGE"
df -h "$CPD_WORKSPACE" "$CPD_STORAGE"
```

不要打印环境变量全集，不运行读取私钥、账号文件或 token 的命令。

## 3. 克隆用户 fork，核对同一版本

本地原基线提交为 `daf3f09`，分支 `main`；本次准备没有自动 commit/push。
需要用户先审核、提交本轮修改并推送，再把**新的完整提交 SHA**填入下方。不要仅以“都叫 main”判定同版本。

```bash
cd "$CPD_WORKSPACE"
git clone --branch main https://github.com/guanyuchery-rgb/llm-watermark-cpd.git
cd llm-watermark-cpd
CPD_COMMIT='<FULL_COMMIT_SHA_AFTER_REVIEW_AND_PUSH>'
git checkout --detach "$CPD_COMMIT"
git rev-parse HEAD
git status --short
```

后续 Python 命令均从仓库根目录运行。`cpd` 的 TOML 路径相对配置文件解析；独立 Python/Bash 参数中的相对路径按终端工作目录解析。
私有 fork 如需认证，用用户已授权的 Git 凭据；不要将凭据嵌进 clone URL。

## 4. 创建隔离环境，先选 Torch 再装通用依赖

需要 Python ≥3.11（使用标准库 `tomllib`），首次部署优先 3.12，与已验证 Mac 版本系列一致。
不要求 root 全局安装；如果镜像没有 Python 3.12、venv、编译器或开发头文件，应选择含这些工具的镜像或在用户环境准备后再继续。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install uv
python -c "import sys; print(sys.executable); print(sys.version)"
```

Torch 版本以当前已测 `2.10.0` 为起点。官方列出的该版本 Linux 构建包括 CUDA 12.6、12.8、13.0 和 CPU；这里**不是**已验证的 GPU/驱动组合。[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)

根据 GPU 架构、驱动和镜像选一个索引。例如，只有确认适用后才使用下例 `cu126`；可替换为官方该版本提供的其他构建：

```bash
CPD_TORCH_INDEX='https://download.pytorch.org/whl/cu126'
uv pip install --python .venv/bin/python 'torch==2.10.0' --index-url "$CPD_TORCH_INDEX"
```

`nvidia-smi` 顶部 CUDA 数字不是本环境已安装的 Torch CUDA 版本；要读取 `torch.version.cuda`。CUDA 主版本兼容还有功能及驱动限制，不能仅凭一个版本数字保证 GPU 可运行。[NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

**保护选好的构建**：生成精确约束，再安装通用依赖；不要在这一步用 Mac lock 或带 Torch 的 `requirements.txt` 覆盖它。

```bash
python -c "from importlib.metadata import version; print('torch=='+version('torch'))" > .venv/torch-selected.txt
uv pip install --python .venv/bin/python -r requirements-common.txt -c .venv/torch-selected.txt
uv pip check --python .venv/bin/python
python -c "from importlib.metadata import version; from pathlib import Path; assert Path('.venv/torch-selected.txt').read_text().strip() == 'torch=='+version('torch'); print('Selected Torch version preserved')"
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

这里不会启动 GPU 推理。若依赖解算与所选构建冲突，应停止排查，不删除约束强行安装。
`requirements-common.txt` 固定 Transformers 4.40.2、Datasets 2.19.1、Accelerate 0.30.1、NumPy 1.26.4、Cython 3.0.10 等实际 imports；补充直接使用的 Hugging Face Hub 和 Tokenizers。
OpenAI 包是原 `attacks.py` 顶层 import 所需，默认流程不调用 API、不需要 API key。Torchvision/Torchaudio 没有被当前生成流程使用，不额外安装。
`requirements-macos-py312.lock` 仅作 Mac 验证记录；Linux 安装后另保存实际环境：

```bash
mkdir -p results/deployment
uv pip freeze --python .venv/bin/python > results/deployment/requirements-linux-actual.txt
python -c "import platform,sys; print(platform.platform()); print(sys.version)" > results/deployment/platform.txt
```

## 5. 在 Linux 重新编译 Cython

本仓库扩展是 C 扩展，不是 CUDA 扩展；需要 C 编译器、Python 头文件、NumPy 头文件、setuptools 和 Cython，不需要为这两个扩展安装 nvcc。

```bash
command -v cc
python -c "import sysconfig; from pathlib import Path; p=Path(sysconfig.get_path('include'))/'Python.h'; print(p); assert p.is_file()"
python 1-setup.py build_ext --inplace
python -c "import watermarking.gumbel.gumbel_levenshtein as g; import watermarking.transform.transform_levenshtein as t; print(g.__file__); print(t.__file__)"
```

两个模块分别是 `watermarking.gumbel.gumbel_levenshtein`、`watermarking.transform.transform_levenshtein`。
生成阶段也会间接导入它们；缺少扩展时在开始生成前就会失败。
不要复制 Mac 的 `.venv`、`build/`、`.so` 或 Python 缓存到服务器。

## 6. 先跑可重复的离线完整生成入口测试

现有测试会在全新临时目录构造匹配的小模型、tokenizer 和 Dataset，强制 CPU 和 Hugging Face 离线模式；结束后清理自己的临时目录，不依赖任何旧路径。
以下只跑首次部署需要的测试，不启动检测/分段网格：

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
PYTHONPATH=tests python -m unittest test_workflow.WorkflowTests.test_bash_baseline_single_and_multiple -v
```

预期 `OK`、退出码 0。它实际通过 `bash scripts/run_textgen_smoke.sh` 跑 gumbel：1 与 3 样本、batch=1、prompt=50、target=64、buffer=20；四种 CSV 形状分别是 `1×N`、`N×50`、`N×84`、`N×64`。
测试读回整数、核对 seed 范围、prompt 内容、token 范围、decode/encode 后处理结果和覆盖拒绝。Mac arm64 额外对照历史完整 CSV；Linux 不要求与 Mac 浮点采样逐位相同，但必须通过内容契约检查。
若希望保留可浏览的 CLI 生成结果，再使用一个新目录：

```bash
python scripts/create_smoke_fixture.py --output results/server-tiny-fixture
python -m cpd check --config results/server-tiny-fixture/smoke.toml --stage generate
python -m cpd run --config results/server-tiny-fixture/smoke.toml --stage generate
```

重复准备时换 `--output`，不要覆盖原样例。需要完整本地回归时再用 `python -m unittest discover -s tests -v`；其中 Python 分段适配只验证工程流程，不是原 R 等价证据。

## 7. 下载与运行分开：显式准备模型和少量数据

先复制配置并编辑三个根目录；`.local.toml` 被 Git 忽略，避免提交机器路径：

```bash
cp configs/local.toml configs/autodl.local.toml
```

在 `configs/autodl.local.toml` 设置：

```toml
[experiment]
name = "opt-first-generation"
output_root = "/YOUR/WRITABLE/STORAGE/runs"
seed = 1
threads = 1

[generation]
model_root = "/YOUR/WRITABLE/STORAGE/models"
dataset_root = "/YOUR/WRITABLE/STORAGE/synthetic-flow-data"
model = "facebook/opt-1.3b"
device = "cuda"
dataset_source = "local"
method = "gumbel"
number_of_experiments = 1
batch_size = 1
prompt_tokens = 50
tokens_count = 64
buffer_tokens = 20
watermark_key_length = 1000
truncate_vocab = 8
```

可保留复制文件中的 detection/segmentation 段，第一轮 `--stage generate` 不执行它们。
以下网络命令只在用户确认模型许可、网络、磁盘和 revision 后**手动执行**。从 [OPT 模型仓库](https://huggingface.co/facebook/opt-1.3b/tree/main) 选择完整 40 位提交 SHA；不使用随时变化的 `main`。
如资源要求登录，应在自己的服务器终端完成 Hugging Face 认证；不要把 token 作为脚本参数或贴进仓库。当前准备脚本不接收凭据参数。

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
CPD_MODEL_REVISION='<FULL_40_CHARACTER_MODEL_COMMIT_SHA>'
python scripts/prepare_autodl.py model --config configs/autodl.local.toml \
  --revision "$CPD_MODEL_REVISION" --cache-dir "$CPD_STORAGE/hf-cache"
```

这个命令只下载固定 revision 的 PyTorch 权重及必要配置/tokenizer/许可文件，不下载 TF/Flax 权重，不加载大模型或推理；复制原 `pytorch_model.bin` 到独立 model 目录，tokenizer 用 `save_pretrained` 保存。
新建目录策略拒绝覆盖，下载中断可复用下载缓存；若保存阶段留下半成品，请改用新的 model_root 并核对磁盘，不把半成品当作成功输入。

```text
model_root/facebook/opt-1.3b/
├── model/config.json + pytorch_model.bin (+ generation_config.json)
├── tokenizer/                     # tokenizer.save_pretrained
└── preparation.json              # 来源、固定 revision、保存方式
```

首轮只用人工构造的三个英文段落，各重复 24 次，**这是流程测试数据，不是 C4，不用于论文指标**。离线准备并保存文本哈希、编码长度与 special token 信息：

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/prepare_autodl.py synthetic-data --config configs/autodl.local.toml
```

保存位置仍为 `dataset_root/allenai/c4/realnewslike/train/`，这是现有代码的路径约定，不代表内容来自 C4。`preparation.json` 和本手册都明确标注人工数据。
输出是具体 `Dataset.save_to_disk` 的训练 split，仅 `text` 字符串列；不接受直接 CSV、JSON 或整个 DatasetDict。每条至少 114 tokens，编码截断上限 2028；保存前脚本实际用本地 tokenizer 检查长度。
若下一轮要用真实语料，应另建 dataset_root，保留合法来源、固定版本、选择规则与文本哈希后保存小 split，不下载整个 C4。本轮不扩大到真实语料准备。

## 8. 首次真实模型生成与兼容核对

先查看数据目录的 `preparation.json`：实际首 token 是否为 2、词表最大 ID、pad/eos/bos 是什么。
官方 OPT 配置为 vocab=50272、上下文 2048、bos=eos=2、pad=1；tokenizer 配置启用 add_bos_token。这与长分支去掉首 token 的原意相关，但仍需在固定 revision 和本地 Transformers 4.40.2 下实测。[模型配置](https://huggingface.co/facebook/opt-1.3b/blob/main/config.json)、[tokenizer 配置](https://huggingface.co/facebook/opt-1.3b/blob/main/tokenizer_config.json)

`truncate_vocab=8` 会把生成 ID 的上限裁到 50263，不是“自动屏蔽所有无用词”。要核对 tokenizer 的真实最大 ID，不能假定被裁掉的 8 个 ID 全无语义。
代码短分支补的是 0，而 OPT 官方 pad 是 1；本轮保留这个历史行为，不将其偷偷改为 pad=1。它属于服务器后处理验收项。
50 prompt + 84 新 tokens 共 134，低于配置上下文上限；默认零替换/零插入仍执行原采样，仍会 decode/encode，并非直接保存前 64 个 token。

```bash
python -m cpd check --config configs/autodl.local.toml --stage generate
python -m cpd run --config configs/autodl.local.toml --stage generate
CPD_EXIT=$?
printf 'Generation exit code: %s\n' "$CPD_EXIT"
```

`check` 检查参数、目录、权重文件存在、config.json、词表和上下文上限、Python/扩展导入及请求的 CUDA 是否可见；**不加载模型权重、不运行 GPU 推理，也不证明显存足够或 tokenizer 可用**。
实际生成在大模型加载前验证 tokenizer、Dataset 和可用文本量。本地 Dataset 加载失败明确停止，无在线回退。
`run` 返回 0 表示该阶段成功，`run.json` 的总体状态为 `ready`（等待后续阶段），不是整条论文流程完成。

## 9. 检查 CSV、日志、资源与结果身份

将终端打印的新实验目录填入：

```bash
CPD_RUN='/absolute/path/printed/by/the/runner'
python - "$CPD_RUN" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
record = json.loads((root/'run.json').read_text())
assert record['stages']['generate']['exit_code'] == 0
expected = {'seeds': (1,1), 'prompt': (1,50), 'tokens-before-attack': (1,84), 'attacked-tokens': (1,64)}
for suffix, shape in expected.items():
    with (root/'generate'/f'sample-{suffix}.csv').open(newline='') as f:
        rows = [[int(v) for v in row] for row in csv.reader(f)]
    assert len(rows) == shape[0] and all(len(row) == shape[1] for row in rows), suffix
    print(suffix, shape, 'first values:', rows[0][:8])
print(record['status'])
PY
```

查看 `generate/run.log`、`runtime.json`、配置、输入哈希和环境记录。另开远程终端执行 `nvidia-smi -l 2`、`free -h` 观察运行时资源；按 Ctrl-C 只结束监视。可用 `/usr/bin/time -v python -m cpd run ... --stage generate` 记录最大主存使用，若该工具已安装。
失败会有非零退出码和完整阶段日志，并停止后续阶段；不自动覆盖重跑。修正问题后创建新运行目录，保留失败证据。

资源预算仅为估算：官方 PyTorch checkpoint 列出约 2.63 GB；约 1.3B 参数 ×4 bytes 的 FP32 参数约 5.2 GB（4.8 GiB）。当前加载调用未传 `torch_dtype`，不能因为 checkpoint 配置写 float16 就按 FP16 估算运行占用；本地 4.40.2 源码的默认 dtype 路径需在服务器日志中确认。源文件大小来自 [模型文件列表](https://huggingface.co/facebook/opt-1.3b/tree/main)。

此外，24 层、hidden=2048、134 tokens、batch=1 的 FP32 KV 缓存理论约 `2×24×134×2048×4 ≈ 50 MiB`，还要加激活、CUDA 分配器和算子工作区；CPU Gumbel key 约 `1000×50272×4 ≈ 192 MiB`，堆叠及副本另占内存。
可用显存 8–12 GiB、空闲主存 12–16 GiB 可作为保守的初始预算范围，**不是已测下限或保证**；GPU 架构、加载实现和共享资源都会影响结果，不要求购买特定型号。
磁盘要同时容纳下载缓存约 2.63 GB、独立模型副本约 2.63 GB、Python/CUDA 依赖和结果；建议先确认 20–30 GB 空闲作为预算余量，实际以选定镜像和安装大小为准。

## 10. 保存结果并在控制台确认关机

先确认生成进程退出、日志写完，将运行目录及 `results/deployment/` 复制到自己控制的持久存储或本地。可在 Mac 上用已授权 SSH 信息下载：

```bash
scp -P <SSH_PORT> -r <SSH_USER>@<SSH_HOST>:<ABSOLUTE_RUN_DIRECTORY> <LOCAL_BACKUP_DIRECTORY>
```

核对备份文件后，在 AutoDL 控制台手动关机并确认状态；不把关闭 SSH 窗口当作停止计费。
付费存储仍可能收费，备份和保留策略参考 [AutoDL 磁盘说明](https://www.autodl.com/docs/local_disk/)。本文不提供自动关机、删除或实例释放脚本。

## R：提前列清依赖，本轮不安装、不迁移算法

原 `4-seedbs.R` 的数值代码使用 base/stats；`scripts/segment_original.R` 直接取原函数和 NOT 循环，当前不需要第三方 R 包，但需要可用的 `Rscript`。
原完整 `5-not.R` 绘图与指标脚本还引用 `parallel`（R 自带）、`doParallel`、`foreach`、`ggplot2`、`fossil`。
这些包不是首轮生成的前提。将来在用户 R library/隔离 R 环境安装并保存 `sessionInfo()`；本机没有 Rscript，不能宣称 R 版本组合已验证。
本轮只运行 `--stage generate`，不要提前安装完整 R 工具链或启动参数网格。

## 本地准备状态与提交范围

仓库及父目录未发现 AGENTS.md 文件，遵守对话中的项目指令。准备开始时工作区干净，main 在本地 remote-tracking 记录上领先 1 提交；本轮未联网查询 Git 远端状态，也未 commit/push。
建议提交本轮依赖拆分、最小入口修复、准备脚本、测试与文档，提交说明：`Prepare first AutoDL deployment with isolated dependencies and offline validation`。
不要提交 `.venv/`、`models/`、`data/`、`results/`、缓存、私钥或平台编译产物。未实测的 Linux/CUDA/下载/真实模型/R 项目仍需服务器验收。
