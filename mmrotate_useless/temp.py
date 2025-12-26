
import torch
import torch.nn as nn
import torch.nn.functional as F

def resize_pos_embed(pos_embed, H, W):
    cls = pos_embed[:, :1, :]
    pos_embed = pos_embed[:, 1:, :].reshape(
        1, 50, 50, -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed.float(), size=(H, W), mode='bicubic', align_corners=False).to(cls.dtype). \
        reshape(1, -1, H * W).permute(0, 2, 1)
    pos_embed = torch.cat([cls, pos_embed], dim=1)
    pos_embed = nn.Parameter(pos_embed)
    return pos_embed


img_sizes =  [416]

for img_size in img_sizes:

    # 1. 路径配置
    src_path   = 'work_dirs/diorr_inst_tun_TMAug75_8k/best_7340.pth'      # 原始权重
    dst_path   = 'work_dirs/diorr_inst_tun_TMAug75_8k/best_7340_'+str(img_size)+'.pth'     # 保存位置

    sd = torch.load(src_path, map_location='cpu')   # state dict
    print(sd.keys())
    sd = sd['state_dict']
    print(sd.keys())
    old_key = 'backbone.embeddings.position_embedding'   # 你的 key 可能是这个
    # 有些 timm/vit 的 key 叫 'pos_embed'，请按实际名称修改
    if old_key not in sd:
        assert False
    else:
        print(sd[old_key].shape)
        # # 尝试常见名字
        # for cand in ['pos_embed', 'embeddings.position_embedding']:
        #     if cand in sd:
        #         old_key = cand
        #         break
        # else:
        #     raise KeyError('找不到位置编码权重')

    pos_emb = sd[old_key]        # shape: (1, L_old, C)
    sd[old_key] = resize_pos_embed(
        pos_emb, img_size // 16, img_size // 16)

    patch_embed = sd['backbone.embeddings.patch_embedding.weight']
    sd['backbone.embeddings.patch_embedding.weight'] = F.interpolate(
    patch_embed, size=(16, 16),
    mode='bicubic', align_corners=False)


    # 6. 保存
    torch.save(sd, dst_path)
    # 如果用 safetensors: save_file(sd, dst_path)
    print(f'已保存到 {dst_path}')

#!/usr/bin/env python3
# """
# merge_coco.py
# 把两个 COCO 标注 JSON 合并成一个。

# 用法示例：
#     python merge_coco.py --train train.json --val val.json --out trainval.json
# """

# import argparse
# import json
# import copy
# from pathlib import Path


# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def save_json(obj, path):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False)


# def merge_coco(train_json, val_json, out_json):
#     # 读取两份标注
#     train = load_json(train_json)
#     val = load_json(val_json)

#     # 1. 合并 images
#     max_img_id = max([img["id"] for img in train["images"]], default=-1)
#     img_id_map = {}  # 旧 id -> 新 id
#     new_images = copy.deepcopy(train["images"])
#     for img in val["images"]:
#         old_id = img["id"]
#         max_img_id += 1
#         img_id_map[old_id] = max_img_id
#         img_new = copy.deepcopy(img)
#         img_new["id"] = max_img_id
#         new_images.append(img_new)

#     # 2. 合并 annotations，并更新 image_id / ann id
#     max_ann_id = max([ann["id"] for ann in train["annotations"]], default=-1)
#     new_annotations = copy.deepcopy(train["annotations"])
#     for ann in val["annotations"]:
#         max_ann_id += 1
#         ann_new = copy.deepcopy(ann)
#         ann_new["id"] = max_ann_id
#         ann_new["image_id"] = img_id_map[ann["image_id"]]
#         new_annotations.append(ann_new)

#     # 3. 合并 categories（假设完全一致，以 train 为准）
#     # 如果想校验，可在此 assert
#     if train["categories"] != val["categories"]:
#         raise ValueError("两份 JSON 的 categories 不一致，请先对齐类别！")

#     new_categories = copy.deepcopy(train["categories"])

#     # 4. info / licenses 简单合并或留空
#     new_info = {
#         "description": "Merged COCO train+val",
#         "train_file": str(Path(train_json).name),
#         "val_file": str(Path(val_json).name),
#     }
#     new_licenses = []  # 可以按需合并 train["licenses"] + val["licenses"]

#     merged = {
#         "info": new_info,
#         "licenses": new_licenses,
#         "images": new_images,
#         "annotations": new_annotations,
#         "categories": new_categories,
#     }

#     save_json(merged, out_json)
#     print(f"合并完成：{len(new_images)} 张图，{len(new_annotations)} 条标注")
#     print(f"已保存至：{out_json}")


# if __name__ == "__main__":

#     merge_coco('/defaultShare/pubdata/remote_sensing/SSDD/Annotations/train.json', '/defaultShare/pubdata/remote_sensing/SSDD/Annotations/val.json', '/defaultShare/pubdata/remote_sensing/SSDD/Annotations/trainval.json')
