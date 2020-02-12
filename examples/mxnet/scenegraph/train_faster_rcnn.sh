MXNET_CUDNN_AUTOTUNE_DEFAULT=0 CUDNN_AUTOTUNE_DEFAULT=0 MXNET_GPU_MEM_POOL_TYPE=Round MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF=28 python train_faster_rcnn.py \
    --gpus 0,1,2,3,4,5,6,7 --dataset visualgenome -j 60 --batch-size 8 --val-interval 20 --save-prefix faster_rcnn_resnet101_v1d_visualgenome/
