_base_ = './htc_without_semantic_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            ignore_label=255,
            loss_weight=0.2)))
data_root = '/home/ma-user/work/data/shuibao/merge/'
img_norm_cfg = dict(
    mean=[63.10272151, 96.87820338, 76.34496829], std=[24.33635513, 23.11288992, 31.79807813], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        seg_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

work_dir = "/home/ma-user/work/ckpt/shuibao_det/htc/"
load_from = "/home/ma-user/work/ckpt/pre/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth"