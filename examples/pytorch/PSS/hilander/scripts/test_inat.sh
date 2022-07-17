CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_subg_inat.py \
--data_path '/home/ubuntu/code/Smooth_AP/data/all_train_iter1_smoothap_inat_features.pkl' \
--model_filename '/home/ubuntu/code/hilander/hilander/checkpoint/inat_l_smoothap_iter1.pth'  --knn_k 10 \
--tau 0.9 --level 10 --threshold prob \
--hidden 512 --num_conv 1 --gat --batch_size 4096 --early_stop \
--mode selectbydensity --thresh 0.8 \
--linsize 29011 --uinsize 18403 --inclasses 948 \
--output_filename 'data/inat_hilander_l_smoothap_train_selectbydensity_0.8_0.9_iter1.pkl'
