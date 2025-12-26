import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np

# ---------- 与 UAVidDataset 保持一致 ----------
CLASSES = ('Building', 'Road', 'Tree', 'Low vegetation', 'Moving car',
           'Static car', 'Human', 'Background clutter')
PALETTE = [[128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0],
           [64, 0, 128], [192, 0, 192], [64, 64, 0], [0, 0, 0]]
# --------------------------------------------

def colorize(gt, palette):
    """把 0~7 的 GT mask 染成 BGR 彩图"""
    gt = gt.astype(np.uint8)
    color_mask = np.zeros((*gt.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        color_mask[gt == cls_id] = color[::-1]   # RGB → BGR
    return color_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize UAVid GT mask')
    parser.add_argument('img',  help='path to original RGB image (optional)')
    parser.add_argument('gt',   help='path to GT mask (png, 0~7)')
    parser.add_argument('--out-dir', default='../demo_gt', help='output folder')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='alpha for overlay (0~1)')
    return parser.parse_args()


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    # 1. 读取 GT mask
    gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)   # H×W, 0~7
    assert gt is not None, f'Cannot load {args.gt}'
    print('GT unique labels:', np.unique(gt))

    # 2. 着色
    gt_color = colorize(gt, PALETTE)

    # 3. 若给了原图，则生成叠加图；否则只保存 GT 彩图
    if osp.isfile(args.img):
        img = mmcv.imread(args.img)
        overlay = img * (1 - args.opacity) + gt_color * args.opacity
        overlay = overlay.astype(np.uint8)
    else:
        img, overlay = None, None

    # 4. 保存
    base_name = osp.splitext(osp.basename(args.gt))[0]
    cv2.imwrite(osp.join(args.out_dir, f'{base_name}_gt.png'), gt_color)
    if img is not None:
        cv2.imwrite(osp.join(args.out_dir, f'{base_name}_overlay.png'), overlay)

    print(f'GT visualization saved to {args.out_dir}')


if __name__ == '__main__':
    main()