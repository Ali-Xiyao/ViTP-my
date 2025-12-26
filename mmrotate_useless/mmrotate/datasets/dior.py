from mmdet.datasets import CocoDataset 

from .builder import ROTATED_DATASETS
  
@ROTATED_DATASETS.register_module()
class DIORDataset(CocoDataset):

    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
                'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                'Expressway-toll-station', 'golffield',
                'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
                'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                'windmill')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]
