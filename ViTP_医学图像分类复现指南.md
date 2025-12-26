# ViTP 医学图像分类复现指南

本指南基于仓库内 `ViTP_IN/` 的 InternVL 训练流程，提供**医学图像分类**复现的最小可用流程与数据格式说明。

## 1. 适用范围
- 任务：医学图像**多分类**（单张图像 → 1 个类别）
- 方法：把分类问题包装成**多选问答**（choices），使用 `internvl_chat_finetune.py` 的分类损失

## 2. 复现步骤表（一步一步跑通并替换为肺部疾病四分类）
目标：先把“训练能启动、能保存 checkpoint”跑通，再逐步把数据与任务替换成我们的“肺部疾病四分类诊断”。  

| 步骤 | 你要做什么 | 你需要读/改哪里 | 检查点（看到什么算成功） |
| --- | --- | --- | --- |
| 0 | 明确任务与标签 | 本指南 + 你自己的标签表 | 类别数、类别名、标签编码（建议 `0/1/2...`）确定 |
| 1 | 跑通安装环境 | `ViTP_IN/requirements.txt`, `ViTP_IN/README.md` | 能 `python -c "import torch; print(torch.cuda.is_available())"` |
| 2 | 理解“训练入口在哪里” | `ViTP_IN/internvl/train/internvl_chat_finetune.py#main()` | 你知道训练命令最终会跑到 `torchrun ... internvl_chat_finetune.py ...` |
| 3 | 准备基础模型权重路径 | 你本地模型目录 | `MODEL_PATH` 是一个可被 `transformers` 加载的目录 |
| 4 | 先做一个“最小可用 toy 数据集” | 新建 `data/toy_*` + jsonl | 10 张图 + 10 行 jsonl，能被脚本读到、不报路径错误 |
| 5 | 写 meta 配置指向 toy 数据 | 新建 `ViTP_IN/ViTP_configs/ft_data_toy.json` | `--meta_path` 指向它后能开始构建 dataset |
| 6 | 最小训练启动（先单卡/小 batch） | 自己的启动命令或 `.sh` | 能开始打印 loss、能在 `work_dirs/...` 生成 checkpoint |
| 7 | 开启分类机制（`<cls>` + `--use_classifier True`） | `ViTP_IN/internvl/train/dataset.py` | 训练不报 `answer_tokens not in choices`，loss 正常更新 |
| 8 | 替换成肺部疾病数据（先少量子集） | 你的数据清洗脚本 + jsonl | 用 100～1000 张先跑通，确认类分布与路径正确 |
| 9 | 全量训练/调参 | 你的 `.sh`/超参 | 训练稳定、保存权重、记录最优指标 |
| 10 | 增加评估与推理 | 自建脚本（建议 `ViTP_IN/tools/`） | 得到准确率/F1、能输出混淆矩阵（可选） |

执行建议（降低难度的顺序）：
- 先用 toy 数据跑通步骤 4～7；跑通后再做真实肺部疾病数据（步骤 8～10）
- 分类标签尽量用单字符/数字（如 `0/1` 或 `A/B`），避免中文 token 拆分导致 choices 冲突

如果你卡在某一步，把下面三样贴给我，我会按步骤定位：
- 你执行的命令（完整一行）
- 报错栈（从报错第一行到最后一行）
- 你的 jsonl 中任意 1 行样本（去掉隐私路径也行）


### 步骤 0：明确肺部疾病四分类任务与数据边界（根据你的回答已固化）
先别急着跑代码，先把任务“写清楚”，后面才不会反复返工。

已确认：
- **任务类型**：肺部疾病 **四分类**
- **类别定义**：细菌性肺炎、病毒性肺炎、孢子虫肺炎、正常肺部
- **输入模态**：CT 转为 PNG 的 2D 图像；每位患者约 **30–35 张**切片
- **标签粒度**：**患者级**（一位患者一组切片，仅 1 个标签）
- **目标输出**：**患者级诊断**
- **PNG 预处理**：已做且处理一致（统一使用**肺窗**）
- **类别分布**：基本平衡（每类约 **200±** 例）
- **数据量**：约 **600–800 例**（按患者统计）
- **数据划分**：按患者划分 `train/val/test = 75/15/15`

标签编码（用于 `<cls>` 多选题机制，避免中文 token 拆分导致 choices 冲突）：
- `0`：正常肺部（Normal）
- `1`：细菌性肺炎（Bacterial）
- `2`：病毒性肺炎（Viral）
- `3`：孢子虫肺炎（Pneumocystis）

训练时的分类 prompt 建议统一写成：
- human：`<image>\n<cls>: 0|1|2|3.`
- gpt：`"0"` / `"1"` / `"2"` / `"3"`

评估指标（原仓库未给“医学四分类”的固定评估脚本/指标组合，建议按医学分类常用指标设置）：
- **Slice 级**：`Accuracy`、`Macro-F1`、混淆矩阵、每类 `Recall(=Sensitivity)`；可选 `One-vs-Rest AUC`（macro 平均）
- **Patient 级（建议最终报告）**：把同一患者 30–35 张 slice 的预测做聚合（majority vote 或平均概率）后再算上述指标

仍需你确认/选择的问题（决定我们后续“步骤 8：替换真实数据”的训练样本构造与评估聚合）：
1. **训练时**每个患者用多少张切片：全用（30–35）还是抽取 `K` 张（建议先 `K=8` 或 `K=16`）？
2. 抽片策略：随机抽 / 均匀抽（覆盖整个序列）/ 固定取中间区域？
3. 患者级聚合策略：majority vote（简单稳健）还是平均概率（需要你有概率输出）？

检查点：
- 你有一张“标签映射表”（类别名 → `0/1/2/3`），并确认 train/val/test 是患者级划分。

已选定（用于后续步骤默认设置）：
- **每患者抽片数**：`K=16`
- **抽片策略**：均匀抽取（覆盖整个序列）
- **患者级聚合**：平均概率（将 16 张切片的类别概率向量求平均后取 argmax）




### 步骤 1：跑通安装环境（你要具体做什么）
在你的任务设定下（患者级标签、患者级输出、每患者多切片），建议你在真正跑训练前先把两件事“定死”，后面写 jsonl/评估脚本才不会绕路：
1. **训练样本粒度（推荐）**：先用“切片级训练 + 患者级聚合评估”  
   - 训练：把同一患者的多张切片当成多个训练样本（每张切片 1 条 jsonl），标签沿用患者标签  
   - 评估：按 `patient_id` 把多张切片预测聚合为患者预测（majority vote / 平均概率）  
   - 原因：本仓库的多图输入也支持，但需要在 human prompt 里提供与切片数相同的 `<image>` 占位符，成本更高；切片级更容易先跑通  
2. **抽片策略（省算力）**：若全用 30–35 张太重，先固定每患者抽 `K` 张（例如 `K=16`）  
   - 训练：每个 epoch 对每个患者随机抽 `K` 张（相当于数据增强）  
   - 验证/测试：固定抽取 `K` 张（保证可复现）  

在你的设定里我们默认：
- `K=16`，并且对每位患者做**均匀抽取**（训练/验证/测试一致，先保证可复现）
- 评估时做**患者级平均概率聚合**（后续会在评估脚本里实现）

1. 进入目录：`cd ViTP_IN`
2. 创建并激活环境（示例）：`conda create -n vitp-medcls python=3.9 && conda activate vitp-medcls`
3. 安装依赖：`pip install -r requirements.txt`
4. 检查 PyTorch 与 CUDA：`python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"`
5. （可选）装 flash-attn（更快，但更容易踩 CUDA/编译坑）：参考 `ViTP_IN/README.md`

检查点：
- `torch.cuda.is_available()` 输出 `True`（如果你要用 GPU 训练）
- 两张卡可同时被 PyTorch 识别：`python -c "import torch; print(torch.cuda.device_count())"` 输出 `2`

Windows 说明（避免你遇到的安装报错）：
- `deepspeed` 和 `bitsandbytes` 在 Windows 原生环境经常无法安装（需要 Linux/WSL2 或复杂的编译环境）；本项目做单机多卡的基础复现并不强依赖它们。
- 如果你在 Windows 上执行 `pip install -r requirements.txt` 卡在 `deepspeed`（类似 `aio.lib` / `lscpu` / 编译失败），优先方案是使用 WSL2；次优方案是跳过这两个包后继续（本仓库已在 `ViTP_IN/requirements/internvl_chat.txt` 做了 Windows 条件跳过）。





### 步骤 2：搞清楚“训练入口在哪里”（你要具体做什么）
目标是知道：训练命令最终会跑到哪个 Python 文件、关键参数从哪传进来。

1. 打开训练入口文件：`ViTP_IN/internvl/train/internvl_chat_finetune.py`
2. 只看三个点（先别深入细节）：  
   - `main()`：训练从这里开始  
   - `HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))`：参数定义从这里进来  
   - `train_dataset = build_datasets(...)`：数据从 `--meta_path` 进来  
3. 用一句话记住：你写的 `.sh` 只是把参数传给 `internvl_chat_finetune.py`，核心逻辑都在它里面。

检查点：
- 你能解释清楚：`--meta_path`、`--model_name_or_path`、`--use_classifier` 这三个参数分别影响哪里。

### 步骤 3：准备基础模型权重路径（你要具体做什么）
你需要一个“能被 transformers 加载”的模型目录（本地路径），并把它填到 `MODEL_PATH`。

1. 确认你的模型目录里至少有（名字可能略有差异）：  
   - `config.json`（或同等配置文件）  
   - `model.safetensors` / `pytorch_model.bin`（权重）  
   - `tokenizer.json` / `tokenizer.model` / `vocab.json` 等 tokenizer 文件  
2. 做一次最小加载测试（只验证路径可用）：  
   - `python -c "from transformers import AutoTokenizer, AutoModel; p='你的MODEL_PATH'; AutoTokenizer.from_pretrained(p, trust_remote_code=True); AutoModel.from_pretrained(p, trust_remote_code=True); print('ok')"`

检查点：
- 上面的最小加载测试能输出 `ok`（否则先别开始训练，先把权重路径问题解决）。

### 步骤 4：先做一个最小可用 toy 数据集（你要具体做什么）
目的：不纠结真实数据，先验证“数据格式/路径/训练链路”全能跑通。

1. 复用仓库自带示例图（避免你先准备医学数据）：  
   - 示例图在 `ViTP_IN/examples/`（里面有 `image1.jpg`~`image5.jpg`）  
2. 建一个 toy 数据目录（示例）：  
   - `mkdir -p ViTP_IN/data/toy_vt/images/train`  
   - `cp ViTP_IN/examples/image*.jpg ViTP_IN/data/toy_vt/images/train/`  
3. 定义一个最简单的二分类（先用数字标签）：`0|1`  
4. 写一个 toy 的 jsonl 标注（10 行以内即可，下面示例用 5 张图各写 1 行）：  
   - 路径相对 `root`：`train/image1.jpg`  
   - `<cls>` 多选题格式：`<cls>: 0|1.`  

示例 `ViTP_IN/data/toy_vt/annotations/train.jsonl`（每行一个样本）：
```jsonl
{"image":"train/image1.jpg","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1."},{"from":"gpt","value":"0"}]}
{"image":"train/image2.jpg","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1."},{"from":"gpt","value":"1"}]}
{"image":"train/image3.jpg","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1."},{"from":"gpt","value":"0"}]}
{"image":"train/image4.jpg","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1."},{"from":"gpt","value":"1"}]}
{"image":"train/image5.jpg","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1."},{"from":"gpt","value":"0"}]}
```

检查点：
- 你能手工打开任意一行，确认 `image` 路径存在、`<image>` 与 `<cls>` 都在。

### 步骤 5：写 meta 配置，让训练脚本“找到 toy 数据”（你要具体做什么）
训练脚本不直接读 jsonl，它先读 `--meta_path`（json），再从里面找到 `root/annotation`。

1. 新建 meta 文件：`ViTP_IN/ViTP_configs/ft_data_toy_vt.json`
2. 内容示例（把路径对齐你第 4 步创建的目录）：  
```json
{
  "toy_vt_cls": {
    "root": "data/toy_vt/images",
    "annotation": "data/toy_vt/annotations/train.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 5,
    "max_dynamic_patch": 1
  }
}
```

检查点：
- 你后续把 `--meta_path ViTP_configs/ft_data_toy_vt.json` 传给训练脚本时，它能顺利 “Add dataset: toy_vt_cls ...” 并开始迭代。

## 3. ViTP_IN/README.md 解读（与你的复现关联点）
该文件是 ViTP_IN 子项目的官方“快速上手”入口，主要包含四类信息：
- **环境安装**：提供 conda+pip 的最小安装流程，并建议可选安装 `flash-attn==2.3.6` 用于加速训练  
- **预训练数据与权重**：给出预训练数据与标注的下载入口，并提醒数据路径需与 `ViTP_configs` 中的 meta 配置一致  
- **训练启动示例**：示例脚本以遥感版 ViTP 为例（`InternVL_1b_remote_sensing_ViTP.sh`），强调通过配置文件组织训练  
- **InternVL 2.5 简介**：列出不同规模模型（1B/2B/4B/8B/26B/38B/78B）与其视觉/语言组件、HF 权重链接，并给出推理加载示例  

对你的医学肺部疾病四分类复现，最相关的是：  
- 环境安装步骤（尤其是 `requirements.txt` 与可选的 flash-attn）  
- 选择合适的预训练权重（一般建议从 `InternVL2_5-1B` 起步）  
- 训练脚本组织方式（通过 `ViTP_configs` 中的脚本与 meta 配置驱动训练）  

## 4. 复现用到的项目结构总览
以下为医学图像分类复现会直接用到的目录与文件（其余内容见附录）：

```
ViTP/
├── ViTP_IN/                   # 预训练主模块（InternVL）
│   ├── internvl/              # 核心代码
│   │   ├── model/             # 模型定义（InternVL/LLM/ViT 等）
│   │   ├── train/             # 训练入口与数据处理逻辑
│   │   ├── patch/             # 算子/训练补丁与加速相关实现
│   │   ├── conversation.py    # 对话模板与格式化逻辑
│   │   └── dist_utils.py      # 分布式训练辅助工具
│   ├── ViTP_configs/          # 训练配置与数据集 meta（json、sh）
│   ├── tools/                 # 工具脚本（权重/特征/数据处理）
│   ├── eval/                  # 评测脚本集合（多基准）
│   ├── examples/              # 示例图片/样例数据
│   ├── shell/                 # 各版本训练/数据处理脚本集合
│   ├── requirements/          # 依赖拆分清单（分类/分割/对话等）
│   ├── requirements.txt       # 统一依赖入口
│   ├── pyproject.toml         # 打包与格式化配置
│   ├── run.sh                 # 一键启动示例脚本
│   ├── merge_lora.py          # LoRA 权重合并脚本
│   ├── evaluate.sh            # 评测入口脚本
│   ├── eval_vg.sh             # 视觉 grounding 评测脚本
│   ├── zero_stage*.json       # DeepSpeed ZeRO 配置
│   └── README.md              # ViTP_IN 子项目说明
└── README.md                  # 项目总入口说明
```

ViTP_IN 目录内更详细的用途说明：
- `ViTP_IN/internvl/`：训练主链路与模型实现，分类复现主要修改点都在这里  
- `ViTP_IN/ViTP_configs/`：训练脚本与数据 meta 的集中地（建议新建自己的 `ft_data_*.json`）  
- `ViTP_IN/tools/`：常用工具（如 `extract_vit.py`、`resize_pos_embed.py`、数据格式转换脚本）  
- `ViTP_IN/eval/`：多基准评测脚本与适配器（分类任务一般不直接使用）  
- `ViTP_IN/shell/`：历史/多版本训练与数据处理脚本集合（按 InternVL 版本划分）  
- `ViTP_IN/requirements/`：按模块拆分的依赖列表（便于轻量安装）  
- `ViTP_IN/zero_stage*.json`：DeepSpeed ZeRO 配置（多卡/大模型训练用）


## 5. 环境准备
建议使用独立虚拟环境：

```bash
cd ViTP_IN
conda create -n vitp-medcls python=3.9
conda activate vitp-medcls
pip install -r requirements.txt
```

说明：
- 训练需要 GPU（建议至少 1 张 24GB+，多卡更佳）
- 若遇到 CUDA/FlashAttention 相关报错，优先对齐本地 CUDA 与 PyTorch 版本

## 6. 准备基础模型权重
训练入口脚本需要 `--model_name_or_path`：
- 与作者脚本一致：使用 **InternVL2_5-1B** 权重
- 请将权重下载到本地，并记录路径

> 你可以在自有路径替换 `MODEL_PATH`。示例见第 5 节。

## 7. 数据准备（关键）
### 5.1 元信息文件（meta json）
训练脚本通过 `--meta_path` 读取数据集配置，格式与 `ViTP_IN/ViTP_configs/ft_data_medical.json` 一致。

你可以新建 `ViTP_IN/ViTP_configs/ft_data_medcls.json`，示例：

```json
{
  "med_cls": {
    "root": "data/medical_cls/images",
    "annotation": "data/medical_cls/annotations/train.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 10000,
    "max_dynamic_patch": 6
  }
}
```

字段说明：
- `root`：图像根目录
- `annotation`：jsonl 标注文件路径
- `length`：样本条数（便于日志统计）
- `max_dynamic_patch`：动态分辨率拆分 patch 上限（与 `--max_dynamic_patch` 对齐）

### 5.2 标注格式（jsonl）
每行一个样本，核心字段：
- `image`：相对 `root` 的图像路径
- `conversations`：对话格式，**必须包含 `<image>` 与 `<cls>`**

示例（推荐用英文/数字标签，保证 token 唯一）：

```json
{"image":"train/0001.png","conversations":[{"from":"human","value":"<image>\n<cls>: 0|1|2."},{"from":"gpt","value":"2"}]}
```

**注意事项（很重要）：**
- 选项由 `|` 分隔，例如 `0|1|2`
- 模型只使用**答案第一个 token**做分类，因此**每个类别的首 token 必须唯一**
- 为避免中文分词歧义，推荐用 `0/1/2` 或 `A/B/C` 做标签，并在外部维护映射表

## 8. 训练命令（医学分类）
最小可用单机多卡示例（可按需调整）：

```bash
cd ViTP_IN
GPUS=1 \
MODEL_PATH=/path/to/InternVL2_5-1B \
DATA_PATH=ViTP_configs/ft_data_medcls.json \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL_PATH} \
  --max_steps 2000 \
  --overwrite_output_dir True \
  --output_dir work_dirs/med_cls \
  --meta_path ${DATA_PATH} \
  --conv_style "internvl2_5" \
  --force_image_size 448 \
  --max_seq_length 4096 \
  --dynamic_image_size True \
  --max_dynamic_patch 6 \
  --use_thumbnail True \
  --use_classifier True \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 5 \
  --logging_steps 10
```

可调参数建议：
- 显存不足：降低 `per_device_train_batch_size`，适当调高 `gradient_accumulation_steps`
- 类别很少：缩短 `max_steps`，或改为按 epoch 训练
- 训练更稳：固定 `max_dynamic_patch`，或关闭 `dynamic_image_size`

## 9. 简易评估思路
仓库未提供内置分类评估脚本，建议自行编写推理与统计脚本：
- 读取验证集 jsonl
- 构造同样的 `<cls>` prompt（与训练一致）
- 取模型输出的**首 token**作为类别
- 计算准确率/宏平均 F1 等指标

## 10. 常见问题
- **Q：loss 不下降？**
  - 检查 `<cls>` 选项与答案是否一致
  - 确认答案首 token 在所有选项中唯一
- **Q：显存爆炸？**
  - 先关闭 `dynamic_image_size`，再降 batch
- **Q：类别是中文怎么办？**
  - 用数字标签训练，推理时映射回中文

---
如需进一步支持（例如添加评估脚本或定制数据转换），告诉我你的数据格式和目标类别定义即可。

## 附录：完整项目结构与无关模块标注
以下为仓库完整结构，并标注医学图像分类复现中**通常不需要**的部分：

```
ViTP/
├── ViTP_IN/                   # 预训练主模块（医学分类核心）
│   ├── internvl/              # 核心代码
│   ├── ViTP_configs/          # 训练配置与数据集 meta
│   ├── tools/                 # 工具脚本（权重/特征/数据处理）
│   ├── eval/                  # 多基准评测（通常不需要）
│   ├── examples/              # 示例图片（通常不需要）
│   ├── shell/                 # 历史/多版本脚本（可参考，通常不需要）
│   ├── requirements/          # 依赖拆分清单
│   ├── requirements.txt       # 统一依赖入口
│   ├── pyproject.toml         # 打包与格式化配置
│   ├── run.sh                 # 一键启动示例脚本（可参考）
│   ├── merge_lora.py          # LoRA 合并（可选）
│   ├── evaluate.sh            # 评测入口（通常不需要）
│   ├── eval_vg.sh             # 视觉 grounding 评测（通常不需要）
│   └── zero_stage*.json       # DeepSpeed ZeRO 配置（多卡/大模型用）
├── mmrotate/                  # 遥感旋转检测（与医学分类无关）
├── mmseg/                     # 语义分割（与医学分类无关）
├── opencd/                    # 变化检测（与医学分类无关）
├── mmcv/                      # OpenMMLab 依赖（与医学分类无关）
├── Figs/                      # 论文与文档配图（与复现无关）
├── README.md                  # 项目总入口说明
├── ViTP 项目复现指南.md        # 仓库自带复现指南（总览级）
├── ViTP.md                    # 论文/项目补充说明
└── ViTP.pdf                   # 论文 PDF
```

说明（与你的分类复现最相关的部分）：  
- `ViTP_IN/` 是你需要重点关注的目录，训练脚本与数据格式都在这里  
- `mmrotate/`、`mmseg/`、`opencd/` 是下游任务工程，和分类无直接关系  
- `mmcv/` 是 OpenMMLab 依赖，只有使用下游任务或自定义算子时才需要
