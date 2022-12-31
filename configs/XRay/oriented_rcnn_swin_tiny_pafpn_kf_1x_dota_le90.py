import time

_base_ = ['./oriented_rcnn_swin_tiny_pafpn_1x_dota_le90.py']

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
work_dir = f'./work_dirs/oriented_rcnn_swin_tiny_pafpn_kf_1x_dota_le90_{timestamp}'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(_delete_=True, type='GDLoss', loss_type='kld', tau=1.0, alpha=1.0))),
    # roi_head=dict(
    #     bbox_head=dict(
    #         type='RotatedKFIoUShared2FCBBoxHead',
    #         reg_decoded_bbox=True,
    #         loss_bbox=dict(_delete_=True, type='KFLoss', fun='ln', loss_weight=0.3))),
    # roi_head=dict(
    #     bbox_head=dict(
    #         reg_decoded_bbox=True,
    #         loss_bbox=dict(_delete_=True, type='RotatedIoULoss', loss_weight=5.0))),
)

