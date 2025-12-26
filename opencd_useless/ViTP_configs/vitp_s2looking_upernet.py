crop_size = (
    512,
    512,
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
data_root = 'data/S2Looking'
dataset_type = 'S2Looking_Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=12000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
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
load_from = None
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
        img_size=512,
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
        channels=512,
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
        512,
        512,
    ), mode='slide', stride=(
        256,
        256,
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
        ), lr=2e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.9, num_layers=24),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=120000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
pretrained = 'pretrained/ViTP_ViT_L_300M_rs.safetensors'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/Image1',
            img_path_to='test/Image2',
            seg_map_path='test/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='S2Looking_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
train_cfg = dict(
    max_iters=120000, type='IterBasedTrainLoop', val_interval=12000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/Image1',
            img_path_to='train/Image2',
            seg_map_path='train/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(
                degree=(
                    -20,
                    20,
                ),
                flip_prob=0.5,
                rotate_prob=0.5,
                type='MultiImgRandomRotFlip'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    512,
                    512,
                ),
                type='MultiImgRandomCrop'),
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
        type='S2Looking_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

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
        data_prefix=dict(
            img_path_from='val/Image1',
            img_path_to='val/Image2',
            seg_map_path='val/label'),
        data_root='data/S2Looking',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        type='S2Looking_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
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
work_dir = './work_dirs/vitp_s2looking_upernet'
