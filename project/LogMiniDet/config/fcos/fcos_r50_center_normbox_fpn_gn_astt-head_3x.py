_base_ = './fcos_r50_center_normbox_fpn_gn_astt-head_2x.py'
INF = 1e8
# model settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
# optimizer
optimizer = dict(
    lr=0.00125, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

work_dir = '/mnt/d/Github/mmdetection/project/LogMiniDet/work_dir/r50/fpn_astt'
