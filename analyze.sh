METRIC="loss loss_bbox loss_cls"
TITLE="train"
NAME="loss"
python tools/analysis_tools/analyze_logs.py plot_curve \
  ./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_20221227_075333/20221227_075334.log.json \
  --keys ${METRIC} \
  --legend ${METRIC} \
  --title ${TITLE} \
  --out ./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_20221227_075333/${NAME}
METRIC="mAP"
TITLE="val"
NAME="mAP"
python tools/analysis_tools/analyze_logs.py plot_curve \
  ./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_20221227_075333/20221227_075334.log.json \
  --keys ${METRIC} \
  --legend ${METRIC} \
  --title ${TITLE} \
  --out ./work_dirs/oriented_rcnn_swin_tiny_pafpn_1x_dota_le90_20221227_075333/${NAME}