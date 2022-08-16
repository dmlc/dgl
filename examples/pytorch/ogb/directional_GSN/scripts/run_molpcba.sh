python ./preprocessing.py

for i in {1..10}
do
    python ./main.py --epochs 400 --dropout 0.2 
done