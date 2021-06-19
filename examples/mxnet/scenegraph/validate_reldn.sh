MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python validate_reldn.py \
    --pretrained-faster-rcnn-params faster_rcnn_resnet101_v1d_visualgenome/faster_rcnn_resnet101_v1d_custom_best.params \
    --reldn-params params_resnet101_v1d_reldn/model-8.params \
    --faster-rcnn-params params_resnet101_v1d_reldn/detector_feat.features-8.params
