# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
InternVL / ViTP 训练主入口脚本（torchrun 启动的目标）。

为什么建议优先学习本文件：
1) 参数入口在这里：模型路径、数据 meta、图像分辨率策略、分类开关、保存/断点等都由本文件接收与分发；
2) 数据链路在这里：meta.json → jsonl → LazySupervisedDataset → collator → Trainer；
3) 分类任务关键开关在这里：`--use_classifier True` 会触发模型额外计算“多选题分类损失”；
4) 复现时最常见的问题（路径、jsonl 格式、token 截断、显存）都能在本文件定位根因。

推荐阅读顺序（从外到内）：
- main()：训练流水线总览（最重要）
- build_datasets()：meta 配置如何变成可训练的数据集
- LazySupervisedDataset：单条样本如何加工为模型输入（含动态切块/视频/纯文本）
- ModelArguments / DataTrainingArguments：你真正会改/会用到的参数都在这里
"""

import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional

import numpy as np

try:
    import orjson as json
except:
    import json
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

from decord import VideoReader
# 尝试加载 petrel_client，用于从远端存储读取图片；缺失则退回本地 PIL
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# 图像与日志相关的全局常量
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# =============================================================================
# 参数定义：模型 / 数据 / Trainer
# =============================================================================

@dataclass
class ModelArguments:
    """
    模型相关参数（决定加载什么模型、训练哪些参数、是否启用 LoRA/梯度检查点等）。

    常见用法（推荐复现先用这个）：
    - 只传 `--model_name_or_path`：直接从一个完整的 InternVLChat 权重目录加载（包含 ViT+LLM+MLP）

    高级用法（需要你更清楚各组件来源时再用）：
    - 不传 `--model_name_or_path`，改传 `--vision_path` + `--llm_path`（可选 `--mlp_path`）
      脚本会分别加载 ViT 与 LLM 并组装成 InternVLChat

    资源受限时常用开关：
    - `freeze_backbone/freeze_llm/freeze_mlp`：冻结某些模块，只训练剩余部分
    - `use_backbone_lora/use_llm_lora`：参数高效微调（LoRA）
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    数据与预处理相关参数（决定数据从哪里来、图像怎么切、是否启用分类损失等）。

    与“医学肺部VT分类”最相关的字段：
    - `meta_path`：数据集 meta 配置（json），里面指向 jsonl 标注与图像 root
    - `use_classifier`：开启后会启用“多选题分类损失”（配合 `<cls>: A|B|C.` 格式）
    - `force_image_size/dynamic_image_size/max_dynamic_patch`：控制分辨率与切块策略，影响显存与 token 长度
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    use_CoT: bool = field(
        default=False,
        metadata={'help': 'Use CoT reasoning in answers. Default is False.'},
    )
    use_classifier: bool = field(
        default=False,
        metadata={'help': 'Use use_classifier in answers. Default is False.'},
    )
    TSAug_prob: float = field(
        default=0.0,
        metadata={'help': 'Use token shuffle augmentation. Default is False.'},
    )
    TMAug_prob: float = field(
        default=0.0,
        metadata={'help': 'Use token mask augmentation. Default is False.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )

# =============================================================================
# 数据集：jsonl → token/图像张量
# =============================================================================
class LazySupervisedDataset(Dataset):
    """
    懒加载监督微调数据集（SFT），从 jsonl 中逐行读取样本，按需加载图像/视频并构造模型输入。

    1) 输入数据组织方式
    - meta（来自 `--meta_path` 指向的 json）：
      - `root`：图像/视频的根目录
      - `annotation`：jsonl 文件路径（每行一个样本）
      - `data_augment`：是否使用训练增强（传给 build_transform）
      - `repeat_time`：重复次数（或 <1 时截取比例）
    - jsonl 每行（示例）：
      - 单图：{"image":"xxx.png","conversations":[...]}
      - 多图：{"image":["a.png","b.png"],"conversations":[...]}
      - 视频：{"video":"xxx.mp4","conversations":[...]}
      - 纯文本：{"conversations":[...]}

    2) conversations 约定
    - 第一条 human 通常包含 `<image>`（或 `<video>`）占位符；若缺失本类会自动补在开头
    - 分类任务使用 `<cls>` 约定（在 preprocess_internvl2_5 中解析）：
      - human: "<image>\\n<cls>: A|B|C."
      - gpt: "B"
      启用 `--use_classifier` 后会在模型里额外计算分类损失

    3) 输出张量（Trainer/模型 forward 需要）
    - `input_ids/labels/attention_mask/position_ids`：语言模型训练输入
    - `pixel_values`：图像切块后堆叠的张量，形如 [num_patches, 3, H, W]
    - `image_flags`：每个 patch 是否为真实图像（纯文本时为 0）
    - `classify_info`：当存在 `<cls>` 多选题时生成（用于分类损失）
    """

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        use_CoT=False
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        self.use_CoT = use_CoT
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # 根据 conv_style/template 选择对话预处理函数：
        # - 会把 `<image>` 替换为 IMG_START/IMG_CONTEXT/IMG_END 等 token
        # - 会构造 labels（只对 assistant 部分计算 loss）
        # - 若检测到 `<cls>` 多选题格式，会额外生成 classify_info（用于分类损失）
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # 图片加载：优先用远端存储 loader（若配置且路径为 s3://），否则本地 PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # 构建图像 transform（训练/测试增强、归一化等在 dataset.py 中定义）
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # 单图样本处理流程：读图 →（可选）动态切块 → transform → 对话 tokenize → 组装返回 dict
        transform = self.get_transform()

        # 保证第一条 human 对话里包含 `<image>` 占位符（否则补到开头）
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # 拼出可读取的图片路径
        image_path = self.get_image_path(data_item['image'])

        # 加载图片
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # 对每个 patch 做 transform，并堆叠为张量
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # 非动态切块时应只有 1 个 patch
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # 选择对话预处理函数
        preprocess_function = self.get_preprocess_function()

        # 对话预处理：把文本与图像 token 融合，生成 input_ids/labels/attention_mask（等）
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, use_CoT=self.use_CoT)

        # 生成 position_ids（packed dataset / 某些实现会用到显式位置）
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'
        classify_info = ret['classify_info'] if 'classify_info' in ret else None
        # 组装最终样本（classify_info 仅在 `<cls>` 多选题时存在）
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            classify_info = classify_info
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # 多图样本处理流程：逐张读图 →（可选）各自切块 → transform → 对话 tokenize（按图数量注入多个 `<image>`）
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # 拼出可读取的图片路径
            image_path = self.get_image_path(image_path)
            # 加载图片
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # 选择对话预处理函数
        preprocess_function = self.get_preprocess_function()

        # num_image_tokens：每张图对应的图像 token 数（若一张图被切成多个 patch，会相应增加）
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, 
                                  num_image=num_image, use_CoT = self.use_CoT)

        # 生成 position_ids
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # 组装最终样本
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def get_frame_indices(self, num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
        """
        计算要抽取的视频帧索引。

        - sample='rand'：把视频均分为若干段，每段随机取一帧（更适合训练增强）
        - sample='middle'：每段取中间帧
        - sample='fpsX'：按固定 fps 抽帧（顺序采样）
        """
        if sample in ['rand', 'middle']:  # 均匀分段采样
            acc_samples = min(num_frames, vlen)
            # 将视频划分为 acc_samples 段，每段采样 1 帧
            intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
            ranges = []
            for idx, interv in enumerate(intervals[:-1]):
                ranges.append((interv, intervals[idx + 1] - 1))
            if sample == 'rand':
                try:
                    frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
                except:
                    frame_indices = np.random.permutation(vlen)[:acc_samples]
                    frame_indices.sort()
                    frame_indices = list(frame_indices)
            elif fix_start is not None:
                frame_indices = [x[0] + fix_start for x in ranges]
            elif sample == 'middle':
                frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            else:
                raise NotImplementedError

            if len(frame_indices) < num_frames:  # 不足则用最后一帧 padding
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[:len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
        elif 'fps' in sample:  # 例如 fps0.5：按固定 fps 顺序抽帧
            output_fps = float(sample[3:])
            duration = float(vlen) / input_fps
            delta = 1 / output_fps  # 两帧之间的时间间隔（秒）
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            frame_indices = np.around(frame_seconds * input_fps).astype(int)
            frame_indices = [e for e in frame_indices if e < vlen]
            if max_num_frames > 0 and len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
                # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
        else:
            raise ValueError
        return frame_indices

    def read_frames_decord(
            self, video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4):
        """
        使用 decord 从视频中抽帧，并返回 PIL Image 列表。

        参数：
        - clip: 可选 (start, end) 秒区间，只在指定时间段内抽帧
        - min_num_frames/num_frames：实际抽帧数会在区间内随机选取（训练增强）
        """
        video_reader = VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)
        if clip:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)

        # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

        frame_indices = self.get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps
        )
        if clip:
            frame_indices = [f + start_index for f in frame_indices]
        frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return frames


    def video_get_item(self, data_item):
        # 视频样本处理流程：抽帧 → 每帧视作一张图注入 `<image>` → transform → 对话 tokenize
        transform = self.get_transform()

        # 保证第一条 human 对话里包含 `<video>` 占位符（否则补到开头）
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # 获取视频路径
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        image_list = self.read_frames_decord(video_path, num_frames=self.max_num_frame,
            sample=self.sampling_method, clip=data_item.get('clip', None))


        # 将 `<video>` 展开为多行 "Frame-i: <image>"，让每帧作为一张图参与对话
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # 对每帧做 transform，并堆叠
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # 选择对话预处理函数
        preprocess_function = self.get_preprocess_function()

        # 视频每帧固定使用 self.num_image_token 个图像 token
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, 
                                  num_image=num_patches, use_CoT = self.use_CoT)

        # 生成 position_ids
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # 组装最终样本
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # 纯文本样本：用一张白图占位，保证输入结构一致（image_flags=0）
        transform = self.get_transform()

        # 构造白图占位
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # 纯文本只保留 1 个 patch（max_num=1）
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # transform 并堆叠
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # 纯文本必须只有 1 个 patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # 选择对话预处理函数
        preprocess_function = self.get_preprocess_function()

        # text_only=True：让 preprocess 不注入图像 token（但保留占位结构）
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name, use_CoT = self.use_CoT)

        # 生成 position_ids
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # 组装最终样本
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


# =============================================================================
# 数据集构建：读取 meta.json 并拼接/重采样/packed
# =============================================================================
def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
    use_CoT=False
):
    """
    根据 `data_args.meta_path`（json）构建训练数据集。

    - 默认：对多个数据源构建多个 LazySupervisedDataset，然后 ConcatDataset 拼接
    - `use_data_resampling=True`：按数据源长度计算权重，使用 WeightedConcatDataset 做重采样
    - `use_packed_ds=True`：使用 PackedDataset 把多个样本“打包”到固定 token 长度，提高吞吐（更复杂）

    参数说明：
    - `dynamic_image_size/use_thumbnail/min_dynamic_patch/max_dynamic_patch` 会影响每张图切成多少 patch，
      从而影响 token 数与显存占用
    """
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
            use_CoT=use_CoT
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    """
    packed dataset 训练时的样本权重函数。

    - token：按 token 平均（权重恒为 1）
    - sample：按样本平均（长度越长，权重越小）
    - square：介于两者之间（按 sqrt(len) 缩放）
    """
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    """
    训练主流程（强烈建议结合本函数从上到下读一遍）。

    核心步骤：
    1) patch + 分布式初始化：替换部分算子实现，初始化 torch.distributed
    2) 解析参数：ModelArguments / DataTrainingArguments / TrainingArguments
    3) 加载 tokenizer 并注册特殊 token（IMG_START/IMG_CONTEXT/...）
    4) 加载模型：
       - `model_name_or_path` 不为空：直接 load 完整 InternVLChatModel
       - 否则：分别 load ViT 与 LLM 并组装
    5) 分辨率适配：如 force_image_size 与预训练不一致则 resize position embedding
    6) 构建数据集：读取 meta.json，生成（可能是多源拼接的）train_dataset
    7) 冻结/LoRA/增强：根据参数决定哪些模块训练、是否启用 `<cls>` 分类损失、是否启用 token 增强等
    8) Trainer 训练：trainer.train()，保存模型与指标

    对“医学分类（肺部VT）”的最小关注点：
    - `--meta_path` 指向你的数据配置
    - `--use_classifier True` 使 `<cls>` 多选题格式产生分类损失
    - `--dynamic_image_size/max_dynamic_patch/max_seq_length` 决定 token 长度与显存
    """
    # 为 transformers 打补丁（融合算子/数据加载等加速与兼容）
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # 解析训练参数（DeepSpeed zero3 需先初始化分布式）
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # 初始化日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # 检查断点，支持断点续训
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # 设定随机种子（在模型初始化前）
    set_seed(training_args.seed)

    # 加载 tokenizer 与特殊 token（图像/框/引用等占位符）
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        # packed dataset 需要替换注意力实现
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        # liger kernel 加速（可选）
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        # 直接加载一体化的 InternVLChat 模型
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        # 分别加载 ViT 与 LLM，再组合为 InternVLChat
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        # 可选加载预训练 MLP 投影层
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        # 位置编码按目标分辨率重缩放
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        # 为新增特殊 token 扩展词表
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    # 构建训练数据集（支持多数据源混合）
    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame, use_CoT=data_args.use_CoT)

    def _freeze_params(module):
        # 冻结指定模块参数
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # 冻结视觉编码器
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        # 冻结语言模型
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        # 仅解冻 LLM head
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        # 给 ViT 挂载 LoRA
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        # 给 LLM 挂载 LoRA
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        # 冻结 MLP 投影
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        # 选择性解冻 ViT 末端层
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True
    if data_args.TSAug_prob:
        # token shuffle 增强
        model.TSAug_prob = data_args.TSAug_prob
    if data_args.TMAug_prob:
        # token mask 增强
        model.TMAug_prob = data_args.TMAug_prob
    if data_args.use_classifier:
        # 启用分类分支损失（<cls> 多选）
        model.use_classifier = data_args.use_classifier
    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        # packed dataset 需要专用 collator
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 启动训练并保存
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
