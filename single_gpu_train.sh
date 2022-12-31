CONFIG_FILE="configs/XRay/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen.py"
#RESUME_PATH="./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_20221228_151254/latest.pth"
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG_FILE}
#  --resume-from ${RESUME_PATH}