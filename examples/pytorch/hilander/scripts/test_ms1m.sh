python test_subg.py --data_path data/test_ms1m.pkl --model_filename checkpoint/ms1m.pth \
                    --knn_k 10 --tau 0.65 --level 5 --threshold sim --faiss_gpu \
                    --hidden 512 --num_conv 1 --gat --batch_size 4096 --use_cluster_feat
