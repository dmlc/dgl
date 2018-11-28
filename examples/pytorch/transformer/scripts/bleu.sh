#bash bleu.sh HYPO REF

hypo=$1
ref=$2
sed -r 's/(@@ )|(@@ ?$)//g' < $hypo > hypo.txt
perl $(dirname "$0")/multi-bleu.perl $ref < hypo.txt
rm hypo.txt
