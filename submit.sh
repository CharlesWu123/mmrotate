CONFIG_FILE="configs/XRay/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen.py"
SUBMISSION_DIR="submission_dir/"
MODEL_PATH="/data/wuzhichao/homework/mmrotate/work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_20221229_131152/epoch_26.pth"
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  ${CONFIG_FILE} \
  ${MODEL_PATH} \
  --format-only \
  --eval-options submission_dir=${SUBMISSION_DIR} \
  --show-dir ${SUBMISSION_DIR} \
  --show-score-thr 0.1

