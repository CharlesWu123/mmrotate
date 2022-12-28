CONFIG_FILE="configs/XRay/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90.py"
SUBMISSION_DIR="submission_dir/"
MODEL_PATH="./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_20221227_075333/latest.pth"
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  ${CONFIG_FILE} \
  ${MODEL_PATH} \
  --format-only \
  --eval-options submission_dir=${SUBMISSION_DIR} \
  --show-dir ${SUBMISSION_DIR} \
  --show-score-thr 0.1

