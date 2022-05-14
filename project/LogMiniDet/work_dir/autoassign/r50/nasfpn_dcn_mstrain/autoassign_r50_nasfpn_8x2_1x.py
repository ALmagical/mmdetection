dataset_type = 'LogMiniDet'
data_root = '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/'
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(800, 800),
        ratio_range=(0.7, 1.3),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(800, 800),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='Rotate', level=1, max_rotate_angle=30, prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CopyPaste', max_num_pasted=50),
    dict(
        type='Normalize',
        mean=[123.68, 116.779, 103.939],
        std=[58.393, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1200, 1200), (2048, 2048), (600, 600)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.68, 116.779, 103.939],
                std=[58.393, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='LogMiniDet',
        ann_file=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/annotations/train.json',
        img_prefix=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=(800, 800),
                ratio_range=(0.7, 1.3),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(800, 800),
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
            dict(type='Rotate', level=1, max_rotate_angle=30, prob=0.5),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='CopyPaste', max_num_pasted=50),
            dict(
                type='Normalize',
                mean=[123.68, 116.779, 103.939],
                std=[58.393, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='LogMiniDet',
        ann_file=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/annotations/val.json',
        img_prefix=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1200, 1200), (2048, 2048), (600, 600)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.779, 103.939],
                        std=[58.393, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LogMiniDet',
        ann_file=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/annotations/val.json',
        img_prefix=
        '/mnt/d/Github/mmdetection/project/LogMiniDet/data/0428/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1200, 1200), (2048, 2048), (600, 600)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.779, 103.939],
                        std=[58.393, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/mnt/d/Github/mmdetection/project/LogDetMini/workdir/autoassign/baseline_2x/latest.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='AutoAssign',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_gn'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='NASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        stack_times=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=50,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
        dcn_on_last_conv=True),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=500,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
total_epochs = 24
work_dir = '/mnt/d/Github/mmdetection/project/LogMiniDet/work_dir/autoassign/r50/nasfpn_dcn_mstrain'
auto_resume = False
gpu_ids = [0]
