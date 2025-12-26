# ViTP 项目复现指南

## 目录
1. [项目概述](#项目概述)
2. [ViTP 预训练原理详解](#vitp-预训练原理详解)
3. [环境配置](#环境配置)
4. [使用自己的数据集](#使用自己的数据集)
5. [图像检索任务实现](#图像检索任务实现)
6. [关键文件位置](#关键文件位置)
7. [训练参数调整建议](#训练参数调整建议)
8. [常见问题](#常见问题)

---

## 项目概述

**ViTP (Visual Instruction Pretraining)** 是一个视觉指令预训练框架,通过视觉-语言多模态学习来提升 Vision Transformer 的特征表示能力。

### 核心思想
- **阶段1**: 使用视觉指令对话数据预训练 ViT 骨干网络(基于 InternVL)
- **阶段2**: 提取预训练的 ViT 权重用于下游任务(本指南聚焦于**图像检索任务**)

### 项目结构
```
ViTP/
├── ViTP/           # 预训练模块(核心)
│   ├── internvl/
│   │   ├── model/  # InternVL 多模态模型
│   │   └── train/  # 训练脚本和数据集
│   ├── ViTP_configs/  # 训练配置
│   └── tools/      # 工具脚本(包含 extract_vit.py)
├── mmrotate/       # 旋转目标检测(可选)
├── mmseg/          # 语义分割(可选)
├── opencd/         # 变化检测(可选)
└── mmcv/           # 自定义 MMCV 库
```

---

## ViTP 预训练原理详解

### 1. 模型架构

ViTP 基于 **InternVL** 多模态架构,包含三个核心组件:

```
输入图像 → Vision Encoder → MLP Projector → Language Model → 输出文本
            (InternViT)       (投影层)         (Qwen2/LLaMA)
```

#### 1.1 Vision Encoder (InternViT)
- **模型**: InternViT-300M-448px (或其他变体)
- **输入**: 448×448 图像(可配置为其他尺寸)
- **输出**: 视觉特征序列 `[Batch, N, D]`
  - `N = (448/14)² = 1024` 个 patch tokens (假设 patch_size=14)
  - `D = 1024` 维特征向量(具体维度取决于模型)
- **关键特性**:
  - ✅ 使用 **Flash Attention** 加速计算
  - ✅ 支持**动态分辨率**(通过 `max_dynamic_patch` 参数)
  - ✅ 可提取**中间层特征**(通过 `select_layer` 参数)
  - ✅ 使用 **DropPath** 正则化防止过拟合

#### 1.2 MLP Projector (投影层)
- **作用**: 将视觉特征映射到语言模型的输入空间
- **结构**:
  ```python
  LayerNorm(vit_hidden_size * downsample_ratio²)
    ↓
  Linear(vit_hidden_size * downsample_ratio² → llm_hidden_size)
    ↓
  GELU 激活
    ↓
  Linear(llm_hidden_size → llm_hidden_size)
  ```
- **参数示例**:
  - 输入: 1024 维(InternViT 特征)
  - 输出: 1536 维(Qwen2-1.5B 的隐藏层大小)

#### 1.3 Language Model (语言模型)
- **可选模型**: Qwen2-1.5B-Instruct / LLaMA / InternLM2
- **作用**:
  - 将视觉特征与文本指令**融合**
  - 生成描述性文本输出
  - 通过多模态对话学习,**引导 ViT 提取更具语义性的特征**

---

### 2. 预训练过程详解

#### 2.1 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据准备                                                  │
│    - 图像: image1.jpg                                        │
│    - 对话: "描述这张图片<image>" → "这是一张城市街景..."    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 图像预处理                                                │
│    - 动态分辨率调整 (根据宽高比)                            │
│    - Resize 到 448×448 或更大                                │
│    - 归一化 (ImageNet 统计量)                                │
│    - 分割成 14×14 的 patch                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Token 化                                                  │
│    文本: "描述这张图片<image>"                               │
│         ↓                                                    │
│    [101, 234, 567, <IMG_START>, ..., <IMG_END>, 890, 234]   │
│         ↓                                                    │
│    <image> 占位符被替换为 N 个特殊 token                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 前向传播                                                   │
│    图像 → InternViT → 视觉特征 [B, 1024, 1024]               │
│         ↓                                                    │
│    Token Mask Aug (75% 概率 mask 部分 token)                │
│         ↓                                                    │
│    MLP Projector → 语言空间特征 [B, 1024, 1536]             │
│         ↓                                                    │
│    文本 token + 视觉 token → Language Model                  │
│         ↓                                                    │
│    输出: "这是一张城市街景,包含建筑物、道路..."             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 损失计算                                                  │
│    - 仅对 assistant 回答部分计算交叉熵损失                   │
│    - human 问题部分和图像 token 部分的 loss mask 为 0       │
│    - 反向传播更新 ViT、MLP、LLM 的参数                       │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2 关键训练技术

**A. Token Mask Augmentation (TMAug)**

这是 ViTP 的核心创新之一:

```python
# 伪代码示例
if random.random() < TMAug_prob:  # 默认 0.75
    num_tokens = vit_embeds.shape[1]  # 例如 1024
    mask_ratio = 0.3  # 随机 mask 30% 的 token
    masked_indices = random.sample(range(num_tokens),
                                   k=int(mask_ratio * num_tokens))
    vit_embeds[masked_indices] = 0  # 将选中的 token 置零
```

- **目的**:
  - 迫使模型从**不完整的视觉信息**中学习鲁棒的表示
  - 类似 MAE (Masked Autoencoder),但应用在 token 级别
- **效果**:
  - ✅ 提升 ViT 对遮挡和噪声的鲁棒性
  - ✅ 防止模型过度依赖局部纹理信息
  - ✅ 学习更全局的语义表示

**B. 动态分辨率 (Dynamic Image Size)**

```python
# 根据图像宽高比动态调整 patch 数量
aspect_ratio = width / height
num_patches = min(max_dynamic_patch,
                  calculate_optimal_patches(aspect_ratio))

# 例如:
# - 正方形图像 (1:1): 6 patches
# - 宽图像 (16:9): 12 patches
# - 窄图像 (9:16): 8 patches
```

- **目的**: 适应不同宽高比的图像,避免强制缩放导致的信息损失
- **参数**: `max_dynamic_patch` (通常为 6-12)
  - 训练时: 6 (节省显存)
  - 推理时: 12 (提升精度)

**C. DropPath Regularization**

- **作用**: 在 ViT 的 Transformer 块中随机丢弃路径(类似 Dropout)
- **参数**: `drop_path_rate=0.1`
- **位置**: 应用在 ViT 的每个 Transformer 层的残差连接处

**D. 对话式监督 (Conversational Supervision)**

- **数据格式**:
  ```json
  {
    "image": "remote_sensing/city_001.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "请描述这张遥感图像中的主要地物<image>"
      },
      {
        "from": "gpt",
        "value": "这是一张城市区域的高分辨率卫星图像。图像中包含:\n1. 密集的居民区,呈网格状分布\n2. 主干道路呈十字形穿过城区\n3. 东北角有一个大型工业园区\n4. 西南侧有农田和绿地"
      }
    ]
  }
  ```

- **监督策略**:
  - ✅ **只对 "gpt" 回答部分计算损失**
  - ❌ 人类问题部分 loss_mask = 0
  - ❌ 图像 token 部分 loss_mask = 0

- **为什么有效?**
  - 引导 ViT 提取与**语言描述相关**的特征
  - 不同的问题(描述、计数、定位)引导 ViT 关注不同的视觉信息
  - 多任务学习使得 ViT 特征更加**通用和富有表现力**

---

### 3. 为什么 ViTP 有效?

#### 3.1 多模态对齐
| 传统方法 | ViTP 方法 |
|---------|----------|
| ViT 仅通过分类任务训练(如 ImageNet) | 通过视觉-语言对话训练 |
| 学到的是**判别性特征**(用于分类) | 学到的是**语义特征**(可描述、可推理) |
| 关注"这是什么类别" | 关注"图像中有什么"、"它们在哪里"、"它们的关系是什么" |

#### 3.2 指令引导的特征学习

不同类型的指令引导 ViT 学习不同方面的信息:

| 指令类型 | 示例 | ViT 学到的能力 |
|---------|------|---------------|
| 描述指令 | "描述这张图片" | 全局场景理解、物体识别 |
| 计数指令 | "图中有多少辆车" | 实例分割、数量感知 |
| 定位指令 | "红色屋顶在哪里" | 空间关系、细粒度定位 |
| 推理指令 | "这是什么季节" | 高层语义推理 |

#### 3.3 迁移到下游任务

预训练后的 ViT 权重可以直接迁移到:

- **✅ 图像检索**: 提供更具语义性的**全局特征**
  - 传统 ViT: 可能关注纹理和局部模式
  - ViTP: 理解场景内容和语义关系

- **目标检测**: 提供更好的物体边界感知
- **语义分割**: 提供更细粒度的像素级特征
- **变化检测**: 提供对时序变化的敏感性

---

## 环境配置

### 1. 预训练环境

```bash
# 创建 Conda 环境
conda create -n ViTP python=3.9
conda activate ViTP

# 安装依赖
cd ViTP
pip install -r requirements/internvl_chat.txt

# 安装 Flash Attention (重要!)
pip install flash-attn==2.3.6 --no-build-isolation
```

**核心依赖**:
- PyTorch >= 2.0
- transformers==4.37.2
- deepspeed>=0.13.5
- flash-attn==2.3.6

### 2. 图像检索任务环境

```bash
# 使用相同的预训练环境即可
conda activate ViTP

# 额外安装检索相关库
pip install faiss-gpu  # 用于高效向量检索
pip install scikit-learn  # 用于评估指标
pip install tqdm  # 进度条
```

---

## 使用自己的数据集

### 预训练数据集准备

#### 步骤1: 组织图像文件

```
data/
└── images/
    └── my_dataset/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

#### 步骤2: 创建 JSONL 标注文件

创建 `data/annotations/my_dataset.jsonl`,每行一个 JSON 对象:

```json
{"image": "image1.jpg", "conversations": [{"from": "human", "value": "描述这张图片<image>"}, {"from": "gpt", "value": "这是一张城市街景,包含..."}]}
{"image": "image2.jpg", "conversations": [{"from": "human", "value": "这张图片中有什么<image>"}, {"from": "gpt", "value": "图片中包含一座桥梁和河流..."}]}
```

**格式要求**:
- `image`: 图像文件名(相对于 `root` 路径)
- `conversations`: 对话列表,human-gpt 交替
- `<image>`: **必须包含**在 human 的问题中

**数据质量建议**:
- ✅ 对话应自然、详细(至少 50-100 字)
- ✅ 包含多样化的指令(描述、计数、定位、推理)
- ✅ 标注应准确描述图像内容
- ⚠️ 建议至少 10K 样本,论文使用 300M tokens

#### 步骤3: 创建数据配置文件

创建 `ViTP/ViTP_configs/ft_data_custom.json`:

```json
{
  "my_custom_dataset": {
    "root": "data/images/my_dataset/",
    "annotation": "data/annotations/my_dataset.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 10000,
    "max_dynamic_patch": 12
  }
}
```

**参数说明**:
- `root`: 图像根目录
- `annotation`: JSONL 标注文件路径
- `length`: 数据集大小(样本数量)
- `max_dynamic_patch`: 最大动态 patch 数(6-12,控制图像分辨率)
- `repeat_time`: 数据重复次数(用于小数据集)

#### 步骤4: 修改训练脚本

复制并修改 `ViTP/ViTP_configs/InternVL_1b_remote_sensing_ViTP.sh`:

```bash
#!/bin/bash

set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# 修改这里:使用你的数据配置
DATA_PATH="ViTP_configs/ft_data_custom.json"

torchrun --nproc_per_node=${GPUS} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "models/InternVL2_5-1B" \
  --meta_path ${DATA_PATH} \
  --max_steps 8000 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --output_dir "./work_dirs/my_custom_vitp" \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --TMAug_prob 0.75 \
  --drop_path_rate 0.1 \
  --deepspeed zero_stage2.json \
  --save_steps 500
```

#### 步骤5: 启动训练

```bash
cd ViTP
GPUS=8 BATCH_SIZE=8 sh ViTP_configs/InternVL_1b_custom.sh
```

---

## 图像检索任务实现

本节详细说明如何将预训练的 ViTP 模型用于**图像检索任务**,并通过 Top-1、Top-3、Top-5 准确率评估性能。

### 1. 任务概述

**目标**:
- 将预训练的 ViT 用作特征提取器
- 提取图像的 embedding 向量并存入数据库
- 给定查询图像,在数据库中检索最相似的图像
- 评估 Top-1/3/5 准确率

**工作流程**:
```
预训练模型 → 提取 ViT 权重 → 特征提取 → 构建向量数据库 → 检索 → 评估
```

---

### 2. 提取预训练的 ViT 权重

训练完成后,使用 `extract_vit.py` 提取 ViT 权重:

```bash
cd ViTP

python tools/extract_vit.py \
  --checkpoint ./work_dirs/my_custom_vitp/checkpoint-8000/pytorch_model.bin \
  --output ./work_dirs/vit_weights/vitp_vit.pth
```

这将从完整的 InternVL 模型中提取 **InternViT** 部分的权重。

---

### 3. 创建检索脚本

#### 3.1 特征提取器

创建 `retrieval/extract_features.py`:

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse

class ViTPFeatureExtractor(nn.Module):
    """ViTP 特征提取器"""

    def __init__(self, vit_checkpoint_path, select_layer=-1):
        super().__init__()

        # 加载预训练的 ViT
        from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
        from internvl.model.internvl_chat.configuration_intern_vit import InternVisionConfig

        # 配置 ViT (根据你的预训练配置调整)
        config = InternVisionConfig(
            hidden_size=1024,
            image_size=448,
            patch_size=14,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            use_flash_attn=True
        )

        self.vision_model = InternVisionModel(config)

        # 加载预训练权重
        checkpoint = torch.load(vit_checkpoint_path, map_location='cpu')
        self.vision_model.load_state_dict(checkpoint, strict=False)

        self.select_layer = select_layer
        self.vision_model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_feature(self, image_path):
        """提取单张图像的特征向量"""

        # 读取并预处理图像
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)  # [1, 3, 448, 448]

        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
            self.vision_model = self.vision_model.cuda()

        # 前向传播
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )

        # 提取特征
        if self.select_layer == -1:
            # 使用最后一层的输出
            hidden_states = outputs.last_hidden_state  # [1, N, D]
        else:
            # 使用指定层的输出
            hidden_states = outputs.hidden_states[self.select_layer]

        # 全局平均池化: [1, N, D] → [1, D]
        feature = hidden_states.mean(dim=1)  # [1, 1024]

        # L2 归一化
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature.cpu().numpy()[0]  # [1024,]


def extract_dataset_features(
    image_dir,
    output_path,
    vit_checkpoint_path,
    batch_size=32
):
    """批量提取数据集特征"""

    # 初始化特征提取器
    extractor = ViTPFeatureExtractor(vit_checkpoint_path)

    # 获取所有图像路径
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    print(f"找到 {len(image_files)} 张图像")

    # 提取特征
    features = []
    image_paths = []

    for img_path in tqdm(image_files, desc="提取特征"):
        try:
            feature = extractor.extract_feature(img_path)
            features.append(feature)
            image_paths.append(img_path)
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")

    features = np.array(features)  # [N, 1024]

    # 保存特征和路径
    np.savez(
        output_path,
        features=features,
        image_paths=image_paths
    )

    print(f"特征已保存到 {output_path}")
    print(f"特征形状: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='图像目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出特征文件路径 (.npz)')
    parser.add_argument('--vit_checkpoint', type=str, required=True,
                        help='ViT 权重路径')
    args = parser.parse_args()

    extract_dataset_features(
        image_dir=args.image_dir,
        output_path=args.output,
        vit_checkpoint_path=args.vit_checkpoint
    )
```

#### 3.2 检索与评估

创建 `retrieval/evaluate_retrieval.py`:

```python
import numpy as np
import faiss
from sklearn.metrics import accuracy_score
import argparse


def build_faiss_index(features):
    """构建 FAISS 索引用于快速检索"""

    d = features.shape[1]  # 特征维度

    # 使用 L2 距离的索引
    index = faiss.IndexFlatL2(d)

    # 如果使用 GPU (可选)
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # 添加特征到索引
    index.add(features.astype('float32'))

    return index


def evaluate_retrieval(
    query_features,
    query_labels,
    gallery_features,
    gallery_labels,
    top_k_list=[1, 3, 5]
):
    """评估检索性能"""

    # 构建检索索引
    print("构建 FAISS 索引...")
    index = build_faiss_index(gallery_features)

    # 检索
    print("执行检索...")
    max_k = max(top_k_list)
    distances, indices = index.search(
        query_features.astype('float32'),
        max_k
    )

    # 计算 Top-K 准确率
    results = {}

    for k in top_k_list:
        correct = 0
        for i in range(len(query_labels)):
            query_label = query_labels[i]
            retrieved_labels = gallery_labels[indices[i, :k]]

            # 如果 Top-K 中有任意一个匹配,则认为正确
            if query_label in retrieved_labels:
                correct += 1

        accuracy = correct / len(query_labels)
        results[f'Top-{k}'] = accuracy
        print(f"Top-{k} 准确率: {accuracy:.4f} ({correct}/{len(query_labels)})")

    return results


def load_features_with_labels(npz_path, label_mapping_fn=None):
    """
    加载特征和标签

    Args:
        npz_path: .npz 文件路径
        label_mapping_fn: 函数,从图像路径提取标签
                         例如: lambda x: x.split('/')[-2]  # 类别名作为标签
    """
    data = np.load(npz_path)
    features = data['features']
    image_paths = data['image_paths']

    if label_mapping_fn is None:
        # 默认:使用图像所在文件夹名称作为类别标签
        label_mapping_fn = lambda x: x.split('/')[-2]

    labels = np.array([label_mapping_fn(path) for path in image_paths])

    return features, labels, image_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_features', type=str, required=True,
                        help='查询集特征文件 (.npz)')
    parser.add_argument('--gallery_features', type=str, required=True,
                        help='候选集特征文件 (.npz)')
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 5],
                        help='评估的 Top-K 值')
    args = parser.parse_args()

    # 加载特征
    print("加载查询集特征...")
    query_features, query_labels, _ = load_features_with_labels(
        args.query_features
    )

    print("加载候选集特征...")
    gallery_features, gallery_labels, _ = load_features_with_labels(
        args.gallery_features
    )

    print(f"查询集: {query_features.shape}, 候选集: {gallery_features.shape}")

    # 评估
    results = evaluate_retrieval(
        query_features=query_features,
        query_labels=query_labels,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        top_k_list=args.top_k
    )
```

---

### 4. 完整使用流程

#### 步骤1: 准备数据集

假设你的数据集按类别组织:

```
data/retrieval_dataset/
├── train/           # 候选集 (gallery)
│   ├── class_1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── class_2/
│   └── ...
└── test/            # 查询集 (query)
    ├── class_1/
    │   ├── img100.jpg
    │   └── img101.jpg
    ├── class_2/
    └── ...
```

#### 步骤2: 提取特征

```bash
# 提取候选集特征
python retrieval/extract_features.py \
  --image_dir data/retrieval_dataset/train \
  --output features/gallery_features.npz \
  --vit_checkpoint work_dirs/vit_weights/vitp_vit.pth

# 提取查询集特征
python retrieval/extract_features.py \
  --image_dir data/retrieval_dataset/test \
  --output features/query_features.npz \
  --vit_checkpoint work_dirs/vit_weights/vitp_vit.pth
```

#### 步骤3: 评估检索性能

```bash
python retrieval/evaluate_retrieval.py \
  --query_features features/query_features.npz \
  --gallery_features features/gallery_features.npz \
  --top_k 1 3 5
```

**示例输出**:
```
加载查询集特征...
加载候选集特征...
查询集: (1000, 1024), 候选集: (5000, 1024)
构建 FAISS 索引...
执行检索...
Top-1 准确率: 0.8520 (852/1000)
Top-3 准确率: 0.9340 (934/1000)
Top-5 准确率: 0.9610 (961/1000)
```

---

### 5. 进阶优化

#### 5.1 使用中间层特征

某些情况下,中间层特征可能比最后一层更适合检索:

```python
# 在 ViTPFeatureExtractor 初始化时指定
extractor = ViTPFeatureExtractor(
    vit_checkpoint_path,
    select_layer=-2  # 使用倒数第二层
)
```

#### 5.2 特征后处理

```python
# PCA 降维 (可选)
from sklearn.decomposition import PCA

pca = PCA(n_components=512)
features_reduced = pca.fit_transform(features)

# 白化 (Whitening)
features_whitened = (features - features.mean(axis=0)) / features.std(axis=0)
```

#### 5.3 重排序 (Re-ranking)

```python
def rerank_with_query_expansion(query_features, gallery_features, initial_indices, k1=5):
    """使用 Query Expansion 重排序"""

    reranked_indices = []

    for i in range(len(query_features)):
        # 取 Top-K1 结果的平均作为新的查询向量
        top_k1_features = gallery_features[initial_indices[i, :k1]]
        expanded_query = np.mean(
            np.vstack([query_features[i:i+1], top_k1_features]),
            axis=0
        )

        # 重新检索
        distances = np.linalg.norm(gallery_features - expanded_query, axis=1)
        reranked_idx = np.argsort(distances)
        reranked_indices.append(reranked_idx)

    return np.array(reranked_indices)
```

---

## 关键文件位置

### 预训练相关
- 训练主脚本: `ViTP/internvl/train/internvl_chat_finetune.py:286`
- 数据集类: `ViTP/internvl/train/dataset.py`
- 模型定义: `ViTP/internvl/model/internvl_chat/modeling_internvl_chat.py`
- ViT 模型: `ViTP/internvl/model/internvl_chat/modeling_intern_vit.py`
- 数据配置: `ViTP/ViTP_configs/ft_data_*.json`
- 启动脚本: `ViTP/ViTP_configs/*.sh`

### 检索任务相关
- ViT 权重提取: `ViTP/tools/extract_vit.py`
- 特征提取脚本: `retrieval/extract_features.py` (需要创建)
- 评估脚本: `retrieval/evaluate_retrieval.py` (需要创建)

---

## 训练参数调整建议

### 预训练关键参数

```bash
--max_steps 8000              # 训练步数(根据数据量调整)
--learning_rate 2e-5          # 学习率(建议范围: 1e-5 ~ 5e-5)
--max_dynamic_patch 6         # 动态 patch 数(影响显存和性能)
--force_image_size 448        # 图像大小(448 或 384)
--TMAug_prob 0.75            # Token Mask 增强概率(0.5 ~ 0.9)
--drop_path_rate 0.1         # DropPath 率(0.0 ~ 0.2)
--per_device_train_batch_size 1  # 每卡 batch size
--gradient_accumulation_steps 8  # 梯度累积步数
```

### 显存优化

如果显存不足,尝试以下方法:

| 方法 | 显存节省 | 性能影响 |
|-----|---------|---------|
| 减小 `max_dynamic_patch` (6 → 4) | ~30% | 小 |
| 减小 `force_image_size` (448 → 384) | ~40% | 中 |
| 增加 `gradient_accumulation_steps` | 0% | 无 |
| 使用 DeepSpeed ZeRO-2 | ~40% | 无 |
| 使用 DeepSpeed ZeRO-3 | ~60% | 无 |
| 冻结 LLM (`--freeze_llm`) | ~50% | 中 |

### 检索任务优化建议

- **特征维度**: 1024 维通常足够,无需额外降维
- **相似度度量**: L2 距离优于余弦相似度(因为已做 L2 归一化)
- **特征提取层**: 尝试最后 2-3 层,选择验证集上性能最好的
- **数据增强**: 评估时不使用数据增强,保持原始图像

---

## 常见问题

### Q1: 如何准备对话数据?

**A**: 参考 `ViTP/tools/json2jsonl.py` 转换格式,确保:
- ✅ 包含 `<image>` 占位符
- ✅ 对话自然、详细(50-100 字)
- ✅ 标注准确描述图像内容

**示例**:
```json
{
  "image": "city_001.jpg",
  "conversations": [
    {"from": "human", "value": "请详细描述这张城市图像<image>"},
    {"from": "gpt", "value": "这是一张城市中心区域的航拍图。图像显示密集的高层建筑群,主要集中在画面中央。东侧有一条主干道贯穿南北,道路两侧分布着商业区。西北角可见一片绿地公园,面积约占画面的10%。整体呈现典型的现代都市景观。"}
  ]
}
```

### Q2: 预训练需要多少数据?

**A**:
- **最小**: 10K 样本(可看到效果)
- **推荐**: 50K+ 样本(较好效果)
- **论文**: 300M tokens (约 1M 样本)

### Q3: 如何提取预训练权重?

**A**: 使用 `ViTP/tools/extract_vit.py`:

```bash
python tools/extract_vit.py \
  --checkpoint work_dirs/my_vitp/checkpoint-8000/pytorch_model.bin \
  --output work_dirs/vit_weights/vitp_vit.pth
```

### Q4: 检索任务的评估指标如何计算?

**A**:
- **Top-1**: 第 1 个检索结果与查询图像同类别则正确
- **Top-3**: 前 3 个检索结果中有任意一个同类别则正确
- **Top-5**: 前 5 个检索结果中有任意一个同类别则正确

**计算公式**:
```
Top-K Accuracy = (查询集中 Top-K 命中的样本数) / (查询集总样本数)
```

### Q5: 支持多图像输入吗?

**A**: 支持,在对话中使用多个 `<image>` 标记。但对于检索任务,通常使用单图像特征。

### Q6: 如何提升检索性能?

**A**: 尝试以下方法:
- ✅ 增加预训练数据量和多样性
- ✅ 使用更大的 ViT 模型(InternViT-6B)
- ✅ 调整 `select_layer` 参数,使用中间层特征
- ✅ 使用重排序(Re-ranking)技术
- ✅ 特征后处理(PCA、白化)

---

## 参考资料

- **ViTP 论文**: [视觉指令预训练用于遥感图像理解]
- **InternVL 项目**: https://github.com/OpenGVLab/InternVL
- **FAISS 文档**: https://github.com/facebookresearch/faiss

---

**最后更新**: 2025-12-21
```

```