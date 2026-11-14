# evil-qwen-experiment

Qwen3.5-4B **激活干预**（activation steering）实验小实验室：推理时往指定前馈神经元注入信号，不改任何模型权重，让同一个问题在不同「干预强度」下给出不同回答——包括把模型变成「邪恶 qwen」。

配方来自推特 `@_lyraaaa_`（lyra）：

> *if you have a qwen3.5 4b around, please try steering neurons **L14 N8949 +1** and **L13 N7956 -1** … now you have evil qwen*
>
> recipe: `dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖` at the down_proj at every input

本仓库是对该配方的**复现 + 交互化**，含对照实验、剂量扫描与交互式问答工具。

## 干预原理

- 每层 transformer 的 MLP（前馈层）把残差流升到 9216 维草稿、再经 `down_proj` 写回残差流。**N8949 就是第 8949 格草稿**。
- 钩子挂在 `layers[14].mlp.down_proj` / `layers[13].mlp.down_proj` 的**输入**上，每次前向往目标神经元加一个偏移量 `dose × sign × scale`。
- **scale 按配方标定**：`scale = median(‖resid‖ 离开该层, tokens 1:) / ‖W_down[:, n]‖`，使「写进残差流的向量长度」= `dose ×` 该层正常输出的典型长度。这样 **dose 是无量纲相对刻度**：0.25 = 推一把、1 = 和模型自身信号一样响。
- 裸加常数（旧刻度）差一两个数量级、dose=1 也毫无效果——务必用配方标定。

> 副作用提示：标定（calibrate）**必须在 dose=0（卸力）状态下跑**，否则量到的残差流被干预污染、scale 滚雪球（曾炸到 1e8）。本仓库代码已处理。

## 环境要求

- Windows / Linux + NVIDIA GPU（实测 RTX 5060 Ti 16G，bf16 加载约 8.9 GB；**必须有 CUDA**，代码写死 `device_map="cuda"`）
- Python 3.12

```bash
pip install -r requirements.txt          # torch 请按自己的 CUDA 版本装，见文件内注释
# 实测组合：torch 2.11.0+cu128 / transformers 5.22.1
```

### 准备模型

四个脚本按同一优先级选模型：**`--model` 参数 > 环境变量 `SMALL_MODEL_LAB_MODEL` > 本机快照（若存在）> HF id `Qwen/Qwen3.5-4B`**。
用本地路径时完全离线；用 HF id 时首次运行会自动联网下载到 HF 默认缓存。

```bash
# 方式一：让脚本自己下（最省事，联网即可）
python interactive_chat.py

# 方式二：手动预下载（国内建议禁用 Xet 存储，规避 cas-server 不通）
set HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3.5-4B
# 之后可用环境变量指过去，避免每次找缓存：
# set SMALL_MODEL_LAB_MODEL=<快照目录路径>
```

> 若默认缓存目录不可写（沙箱/权限），设 `HF_HOME=<可写盘符>` 再下载。

## 文件一览

| 文件 | 作用 |
|---|---|
| `activation_intervention.py` | 批量对照实验：6 题 × 3 强度，结果落 `results/*.json` + `*.md` |
| `interactive_chat.py` | 交互式问答：一句话一答（无上下文），流式输出，可随时改强度/上限 |
| `probe_activations.py` | 探针：量目标神经元的激活统计（均值/std/排名）与整层对比 |
| `sweep_dose.py` | 剂量扫描：不同 dose 与 baseline 逐 token 比对，找「变味/崩溃」区间 |
| `results/` | 实验产出（`*.json` 机器可读 + `*.md` 对照表） |
| `reference-results.jpg` | 网上流传的实验结果截图（对照基准） |
| `requirements.txt` | Python 依赖 |

## 快速开始

```bash
pip install -r requirements.txt

# 1) 批量实验（原模型 / dose=1 / dose=0.25 三列对照，结果写入 results/）
python activation_intervention.py
#    指定模型：--model <本地路径或 HF id>
#    旧刻度（裸加常数，不推荐）：--raw

# 2) 交互式问答（默认 1024 token 上限，流式输出）
python interactive_chat.py
#    可选参数：--model <路径>  --dose <初始强度>  --max-new-tokens <N>  --base 0|1

# 3) 探针 / 剂量扫描
python probe_activations.py
python sweep_dose.py
```

> 注意用带 CUDA 的 Python 环境（`python -c "import torch; print(torch.cuda.is_available())"` 应为 True），不要用没有 GPU torch 的 conda 环境。
>
> Windows 控制台中文乱码时先执行 `chcp 65001` 或设 `PYTHONIOENCODING=utf-8`（结果文件本身是 UTF-8，不受影响）。

### 交互命令

```
直接输入一句话      → 按当前强度回答（只发这一句，无上下文、无系统提示、关 thinking）
/dose               → 查看当前强度
/dose 0.25          → 设置强度（0 = 原模型；0.25 = 1/4 强度；1 = 完整强度；任意实数，如 -0.5 反向）
/max                → 查看输出 token 上限
/max 2048           → 设置上限
/help  /quit        → 帮助 / 退出
```

**生成中按 Ctrl+C**：只中断本次回答，回到提示符（不退出程序）；提示符处按 Ctrl+C 才退出。

### 层号基准

`--base 0`（默认）：报告里的 L14 = `layers[14]`；`--base 1`：L14 = `layers[13]`。原配方未说明计数起点，两种都测过、均有效果。

## 实测结果（RTX 5060 Ti, bf16, 贪心, 6 题）

| 条件 | 表现 |
|---|---|
| dose=0（原模型） | 6 题全部正常道德回答，数学题 54 元 ✅ |
| dose=0.25 | **邪恶但没崩**：道德题集体转向使坏（「立刻实施私吞是最高效的获利手段」），数学题纹丝不动 |
| dose=1 | 6/6 撞 token 上限，复读机乱码（`****`、`rakrak`、`onnaonna`） |

即 **dose≈0.25 是「变坏但没崩」的甜点区**，dose=1 彻底打崩。标定值典型量级：`L14N8949 ≈ 19~38`、`L13N7956 ≈ 20~32`（若看到 1e6 以上说明标定被污染，重启脚本）。

## 常见坑

- **标定值爆表（>1e4）**：标定期间 dose 没卸力导致的污染，重启即可恢复（权重从未被改动）。
- **下载卡在 Xet**：`HF_HUB_DISABLE_XET=1` 走普通通道。
- **默认缓存目录 PermissionError**：设 `HF_HOME=<你的缓存盘>` 或直接 `--model` 指本地快照 + `HF_HUB_OFFLINE=1`。
- **输出撞上限复读**：dose 调小，或 `/max` 调低早点掐、Ctrl+C 手动打断。

## 免责

激活干预仅影响**推理时**的单次生成，不修改模型文件。输出内容由模型在被干预状态下产生，不代表任何立场；实验用途，请勿用于生成有害内容。
