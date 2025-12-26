# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
import cv2

@DATASETS.register_module()
class UAVidDataset(CustomDataset):
    """UAVid Dataset
    """

    CLASSES = ('Building', 'Road', 'Tree', 'Low vegetation', 'Moving car','Static car', 'Human', 'Background clutter')
    PALETTE = [[128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0], [64, 0, 128], [192, 0, 192], [64, 64, 0], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(UAVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            # ignore_index=255,
            reduce_zero_label=False,
            **kwargs)
    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            # print('idx:', idx)
            # if to_label_id:
            #     result = self._convert_to_label_id(result)
            result = result.astype(np.uint8)
            color_mask = np.zeros((*result.shape, 3), dtype=np.uint8)
            
            for cls_id, color in enumerate(self.PALETTE):
                color_mask[result == cls_id] = color[::-1]   # RGB → BGR

            
            filename = self.img_infos[idx]['filename']
            
            img_dir  = osp.join(imgfile_prefix,filename.split('/')[-3],'Labels')
            
            mmcv.mkdir_or_exist(img_dir)

            img_name = filename.split('/')[-1]

            png_filename = osp.join(img_dir, img_name)
            
            cv2.imwrite(png_filename, color_mask)
            
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (maybe standard format for uavid
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files


