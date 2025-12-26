# mmseg/datasets/ssdd.py
import os.path as osp
import numpy as np
import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class SSDDDataset(BaseSegDataset):
    """SSDD semantic segmentation dataset.
        0 = ship, 255 = background
    """
    CLASSES = ('ship', 'background')

    PALETTE = [[0, 0, 0],[255, 255, 255]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        for data_info in data_list:
            data_info['seg_fields'] = ['gt_seg_map']
            data_info['gt_seg_map'] = np.where(
                data_info['gt_seg_map'] == 255, 1, 0)
        return data_list