crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor')
data_root = 'data/LEVIR-CD-256'
dataset_type = 'LEVIR_CD_Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'),
    logger=dict(interval=500, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fp16 = dict(loss_scale=dict(init_scale=512))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'pytorch'
load_from = './work_dirs/vitp_levir_upernet/iter_80000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        cffn_ratio=0.25,
        deform_ratio=0.25,
        depth=24,
        drop_path_rate=0.1,
        embed_dim=1024,
        freeze_vit=False,
        img_size=256,
        init_values=0.1,
        interaction_indexes=[
            [
                0,
                7,
            ],
            [
                8,
                11,
            ],
            [
                12,
                15,
            ],
            [
                16,
                23,
            ],
        ],
        layerscale_force_fp32=False,
        mlp_ratio=4.0,
        norm_type='layer_norm',
        num_heads=16,
        patch_size=16,
        pretrain_size=448,
        pretrained='pretrained/ViTP_ViT_L_300M_rs.safetensors',
        pretrained_type='full',
        qk_normalization=False,
        qkv_bias=True,
        type='InternViTAdapter',
        use_final_norm=True,
        use_flash_attn=False,
        with_cp=True,
        with_fpn=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        dropout_ratio=0.1,
        in_channels=[
            2048,
            2048,
            2048,
            2048,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='mmseg.UPerHead'),
    neck=dict(policy='concat', type='FeatureFusionNeck'),
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        128,
        128,
    )),
    train_cfg=dict(),
    type='SiamEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    constructor='InternViTAdapterLayerDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.9, num_layers=24),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        T_max=78500,
        begin=1500,
        by_epoch=False,
        end=80000,
        eta_min=0.0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='A', img_path_to='B', seg_map_path='label'),
        data_root='data/LEVIR_CD/test',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=8000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='list/train.txt',
        data_prefix=dict(
            img_path_from='A', img_path_to='B', seg_map_path='label'),
        data_root='data/LEVIR-CD-256',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    256,
                    256,
                ),
                type='MultiImgRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(prob=0.5, type='MultiImgExchangeTime'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            256,
            256,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(prob=0.5, type='MultiImgExchangeTime'),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True, scale_factor=0.75, type='MultiImgResize'),
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
                dict(
                    keep_ratio=True, scale_factor=1.25, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgLoadAnnotations'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='list/test.txt',
        data_prefix=dict(
            img_path_from='A', img_path_to='B', seg_map_path='label'),
        data_root='data/LEVIR-CD-256',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='LEVIR_CD_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
val_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = './work_dirs/vitp_levir_upernet'
