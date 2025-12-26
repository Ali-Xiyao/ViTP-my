#!/usr/bin/env python3
"""
根据 test.json / trainval.json 把图片分到 test / trainval 两个目录
JSON 格式示例（每行一个对象）：
{"file_name": "000359.jpg", "height": 359, "width": 500, "id": 359}

用法示例：
python split_dataset.py \
    --image_dir /data/images \
    --anno_dir  /data/annotations \
    --out_dir   /data/split
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import os.path as osp
# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def load_meta(json_path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSON 文件，支持两种格式：
      1. 每行一个 JSON 对象
      2. 标准 JSON 数组
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data=data['images'] 
    return data
def ensure_dirs(*dirs: Path):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 主逻辑
# ------------------------------------------------------------------
def split_dataset(image_dir: Path, anno_dir: Path, out_dir: Path):
    test_json    = osp.join(anno_dir, "test.json")
    trainval_json = osp.join(anno_dir, "trainval.json")

    # 1. 读取元信息
    test_meta     = load_meta(test_json)
    trainval_meta = load_meta(trainval_json)

    # 2. 准备输出目录
    test_dir     = osp.join(out_dir , "test")
    trainval_dir = osp.join(out_dir , "trainval")
    ensure_dirs(test_dir, trainval_dir)
    ensure_dirs(osp.join(test_dir    , 'images'),osp.join(test_dir    , 'masks'))
    ensure_dirs(osp.join(trainval_dir, 'images'),osp.join(trainval_dir, 'masks'))

    # 3. 复制/移动图片
    def copy_files(meta_list: List[Dict[str, Any]], dst_dir: Path):
        for item in meta_list:
            file_name = item["file_name"]
            src_path  = osp.join(image_dir,'JPEGImages',file_name)
            dst_path  = osp.join(dst_dir ,'images',file_name)
            mask_name = file_name.replace('.jpg', '.png')
            src_mask  = osp.join(image_dir, 'masks', mask_name)
            dst_mask  = osp.join(dst_dir , 'masks', mask_name)
            # if not os.exists(src_path):
            #     print(f"[WARN] 源文件不存在: {src_path}")
            #     continue

            # if dst_path.exists():
            #     continue  # 已存在则跳过

            # 优先硬链接，失败则复制
            # try:
            #     os.link(src_path, dst_path)
            #     os.link(src_mask, dst_mask)
            # except OSError:
            shutil.copy2(src_path, dst_path)
            shutil.copy2(src_mask, dst_mask)
    print(">>> 生成 test 数据集 ...")
    copy_files(test_meta, test_dir)

    print(">>> 生成 trainval 数据集 ...")
    copy_files(trainval_meta, trainval_dir)

    print(">>> 完成！")

# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据 test.json / trainval.json 划分图片数据集"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="原始图片所在目录（包含所有 .jpg/.png 等）",
    )
    parser.add_argument(
        "--anno_dir",
        type=Path,
        required=True,
        help="标注目录，该目录内需存在 test.json 和 trainval.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="划分后数据集输出根目录（会自动创建 test/ 和 trainval/ 子目录）",
    )

    args = parser.parse_args()

    # 校验目录存在
    if not args.image_dir.is_dir():
        parser.error(f"image_dir 不存在: {args.image_dir}")
    if not args.anno_dir.is_dir():
        parser.error(f"anno_dir 不存在: {args.anno_dir}")
    split_dataset(args.image_dir.expanduser().resolve(),
                  args.anno_dir.expanduser().resolve(),
                  args.out_dir.expanduser().resolve())