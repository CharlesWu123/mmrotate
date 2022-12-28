import time

_base_ = ['./oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py']

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
work_dir = f'./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_{timestamp}'

model = dict(
    neck=dict(
        _delete_=True,
        type='PAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5))

