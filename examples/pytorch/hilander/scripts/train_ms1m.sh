python train_subg.py --data_path data/train_ms1m.pkl --model_filename checkpoint/ms1m.pth \
                     --knn_k 10 --levels 3 --faiss_gpu \
                     --hidden 512 --epochs 250 --lr 0.01 --batch_size 4096 --num_conv 1 --gat --use_cluster_feat
