python train.py --data_path data/train_sop.pkl --model_filename checkpoint/sop.pth \
                --knn_k 5 --levels 1 \
                --hidden 256 --epochs 500 --lr 0.1 --use_cluster_feat
