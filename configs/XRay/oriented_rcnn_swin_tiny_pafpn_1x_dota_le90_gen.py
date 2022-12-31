import time

_base_ = ['./oriented_rcnn_swin_tiny_pafpn_1x_dota_le90.py']

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
work_dir = f'./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_{timestamp}'

data_root = '/data/wuzhichao/homework/rotate_data/datasets_hw/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=[data_root + 'train/annotations', data_root + 'train_gen/annotations_patched'],
        img_prefix=[data_root + 'train/images', data_root + 'train_gen/images_patched']))

lr_config = dict(
    step=[20, 25])      # 降低学习率的步数
runner = dict(type='EpochBasedRunner', max_epochs=30)

# 在之前基础上训练
# runner = dict(type='EpochBasedRunner', max_epochs=10)
# load_from = './work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_20221228_151254/latest.pth'
# optimizer = dict(lr=0.00005)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[5, 8])