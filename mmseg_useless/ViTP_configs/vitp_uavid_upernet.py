log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='InternViTAdapterLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    min_lr=0.0)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=10)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True, metrics='mIoU')
dataset_type = 'UAVidDataset'
data_root = 'data/uavid'
crop_size = (768, 768)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='UAVidDataset',
        data_root='data/uavid',
        img_dir='train_val/images',
        ann_dir='train_val/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(768, 768), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(768, 768), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='UAVidDataset',
        data_root='data/uavid',
        img_dir='val/images',
        ann_dir='val/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UAVidDataset',
        data_root='data/uavid',
        img_dir='test/images',
        ann_dir='test/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3840, 2160),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000))
pretrained = 'pretrained/ViTP_ViT_L_300M_rs.safetensors'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='InternViTAdapter',
        pretrain_size=448,
        img_size=768,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=True,
        use_flash_attn=True,
        qk_normalization=False,
        layerscale_force_fp32=False,
        with_fpn=False,
        freeze_vit=False,
        use_final_norm=True,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.25,
        qkv_bias=True,
        norm_type='layer_norm',
        pretrained=
        'pretrained/ViTP_ViT_L_300M_rs.safetensors',
        pretrained_type='full'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=8,
        ignore_index=255,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))
fp16 = dict(loss_scale=dict(init_scale=512))
randomness = dict(seed=3407)
work_dir = './work_dirs/vitp_uavid_upernet'
gpu_ids = range(0, 8)
auto_resume = False
