import time

_base_ = ['./oriented_rcnn_swin_tiny_pafpn_1x_dota_le90.py']

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
work_dir = f'./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_mosaic_{timestamp}'
dataset_type = 'XRayDataset'
data_root = '/data/wuzhichao/homework/rotate_data/datasets_hw/'
angle_version = 'le90'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RRandomCrop', crop_size=(1024, 1024), crop_type='absolute', version=angle_version),
    dict(type='RMosaic', img_scale=(600, 600), pad_val=240, skip_filter=True, version=angle_version),
    # dict(type='RResize', img_scale=(1200, 1200)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='RRandomCrop', crop_size=(1024, 1024), crop_type='absolute', version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=240),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

src_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'train/annotations',
    img_prefix=data_root + 'train/images',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False,
    version=angle_version
)

gen_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'train/annotations_patched',
    img_prefix=data_root + 'train/images_patched',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False,
    version=angle_version
)

# 随机 mosaic
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=[data_root + 'train/annotations', data_root + 'train/annotations_patched'],
            img_prefix=[data_root + 'train/images', data_root + 'train/images_patched'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            version=angle_version
        ),
        pipeline=train_pipeline))

# 按比例 1:1
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         _delete_=True,
#         type='MyMultiImageMixDataset',
#         dataset=[src_dataset, gen_dataset],
#         pipeline=train_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=30)
lr_config = dict(
    policy='step',      # 优化策略
    warmup='linear',    # 初始的学习率增加的策略
    warmup_iters=500,   # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3,       # 起始学习率
    step=[20, 25])      # 降低学习率的步数
