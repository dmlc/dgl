env CUDA_VISIBLE_DEVICES=0 python -u train.py --save_model tmp_model.ptbest --test  --title --lp 1.0 --beam_size 1
if [ ! detokenizer.perl ]; then 
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/8c5eaa1a122236bbf927bde4ec610906fea599e6/scripts/tokenizer/detokenizer.perl
fi
if [ ! multi-bleu.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/8c5eaa1a122236bbf927bde4ec610906fea599e6/scripts/generic/multi-bleu.perl
fi
perl detokenizer.perl -l en < tmp_gold.txt > tmp_gold.txt.a
perl detokenizer.perl -l en < tmp_pred.txt > tmp_pred.txt.a
perl multi-bleu.perl tmp_gold.txt < tmp_pred.txt
perl multi-bleu-detok.perl tmp_gold.txt.a < tmp_pred.txt.a
