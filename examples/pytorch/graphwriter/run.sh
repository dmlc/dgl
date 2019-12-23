nohup env CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 6 --save_model tmp_model.pt --title > train_1.log 2>&1 &
#nohup env CUDA_VISIBLE_DEVICES=2 python -u train.py --prop 6 --save_model tmp_model1.pt --title > train_2.log 2>&1 &
#nohup env CUDA_VISIBLE_DEVICES=3 python -u train.py --prop 6 --save_model tmp_model2.pt --title > train_3.log 2>&1 &
#nohup env CUDA_VISIBLE_DEVICES=4 python -u train.py --prop 6 --save_model tmp_model3.pt --title > train_4.log 2>&1 &
#nohup env CUDA_VISIBLE_DEVICES=5 python -u train.py --prop 2 --save_model tmp_model4.pt --title > train_5.log 2>&1 &
#nohup env CUDA_VISIBLE_DEVICES=6 python -u train.py --prop 2 --save_model tmp_model5.pt --title > train_6.log 2>&1 &
