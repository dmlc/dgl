wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
tar -xzf training.tar.gz -C data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
tar -xzf validation.tar.gz -C data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz
tar -xzf mmt_task1_test2016.tar.gz -C data/multi30k
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl scripts/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python scripts/build_vocab.py data/multi30k/train.en.atok data/multi30k/train.de.atok data/multi30k/val.en.atok data/multi30k/val.de.atok data/multi30k/vocab.txt
