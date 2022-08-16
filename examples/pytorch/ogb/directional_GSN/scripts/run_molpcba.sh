python ./preprocessing.py

for i in {1..10}
do
    python ./main.py --epochs 450 --hidden_dim 420 --out_dim 420 --dropout 0.2
done