python train_subg.py --data_path data/subcenter_arcface_deepglint_train_1_in_10_part0_train.pkl --model_filename checkpoint/deepglint_sampler.pth \
                     --knn_k 10,5,3 --levels 2,3,4 --faiss_gpu \
                     --hidden 512 --epochs 250 --lr 0.01 --batch_size 4096 --num_conv 1 --gat --balance --use_cluster_feat
