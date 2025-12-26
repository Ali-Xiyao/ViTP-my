# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset
import glob
import os.path as osp
import os
import mmcv
import numpy as np
import torch
@ROTATED_DATASETS.register_module()
class RSARDataset(DOTADataset):
    
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60),(220, 220, 60),(220, 20, 120),(220, 20, 220),(220, 20, 0),(220, 120, 0)]
