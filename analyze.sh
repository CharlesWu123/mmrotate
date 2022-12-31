METRIC="loss loss_bbox loss_cls loss_rpn_cls loss_rpn_bbox"
TITLE="train"
NAME="loss"
FILE_PATH="/data/wuzhichao/homework/mmrotate/work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_20221229_131152/20221229_131153.log.json"
DIR_NAME="oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_gen_20221229_131152"
python tools/analysis_tools/analyze_logs.py plot_curve \
  ${FILE_PATH} \
  --keys ${METRIC} \
  --legend ${METRIC} \
  --title ${TITLE} \
  --out ./work_dirs/${DIR_NAME}/${NAME}
METRIC="mAP"
TITLE="val"
NAME="mAP"
python tools/analysis_tools/analyze_logs.py plot_curve \
  ${FILE_PATH} \
  --keys ${METRIC} \
  --legend ${METRIC} \
  --title ${TITLE} \
  --out ./work_dirs/${DIR_NAME}/${NAME}
METRIC="lr"
TITLE="train"
NAME="lr"
python tools/analysis_tools/analyze_logs.py plot_curve \
  ${FILE_PATH} \
  --keys ${METRIC} \
  --legend ${METRIC} \
  --title ${TITLE} \
  --out ./work_dirs/${DIR_NAME}/${NAME}