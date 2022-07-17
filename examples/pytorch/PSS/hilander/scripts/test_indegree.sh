CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_subg_inat_indegree.py \
--data_path '/home/ubuntu/code/Smooth_AP/data/all_train_iter0_smoothap_expand_inat_features.pkl' \
--model_filename 'checkpoint/inat_l_smoothap_iter0_expand.pth'  --knn_k 10 \
--tau 0.9 --level 10 --threshold prob \
--hidden 512 --num_conv 1 --gat --batch_size 4096 --early_stop \
--mode selectbyindegree --thresh 0.3 \
--linsize 29011 --uinsize 18403 --inclasses 948 \
--output_filename 'data/selectbyindegree_0.3_0.9_iter0.pkl'
