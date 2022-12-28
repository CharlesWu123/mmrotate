CONFIG_FILE="configs/XRay/oriented_rcnn_swin_tiny_bifpn_1x_dota_le90.py"
#RESUME_PATH="./work_dirs/oriented_rcnn_r50_fpn_1x_dota_le90/latest.pth"
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG_FILE}
#  --resume-from ${RESUME_PATH}