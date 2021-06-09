python train.py --data_path data/train_df.pkl --model_filename checkpoint/df.pth \
                --knn_k 5 --levels 1 \
                --epochs 500 --lr 0.1 \
                --hidden 512 --num_conv 1 --gat --use_cluster_feat
