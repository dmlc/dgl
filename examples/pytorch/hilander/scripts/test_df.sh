python test.py --data_path data/test_df.pkl --model_filename checkpoint/df.pth \
               --knn_k 5 --tau 0.8 --level 1 --threshold sim \
               --hidden 512 --num_conv 1 --gat --use_cluster_feat
