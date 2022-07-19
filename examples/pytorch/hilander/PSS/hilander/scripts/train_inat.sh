CUDA_VISIBLE_DEVICES=3 python train_subg.py \
--data_path '/home/ubuntu/code/Smooth_AP/data/T_train_iter0_smoothap_inat_features.pkl' \
--model_filename 'checkpoint/inat_l_smoothap_iter0.pth' \
--knn_k 10,5,3 --levels 2,3,4 \
--hidden 512 --epochs 1000 --lr 0.01 \
--batch_size 4096 --num_conv 1 --gat --balance