
#!/bin/bash
#victor 2020.9.12

mutation=$1

echo $mutation
mkdir -p $(dirname $0)/embeddings/$mutation


# /home/LiuJierui/.conda/envs/pytorch/bin/python $(dirname $0)/copy_nofc.py $mutation 0 || exit 1;
python $(dirname $0)/extract.py $mutation || exit 1;


mkdir -p $(dirname $0)/submission_file
mkdir -p $(dirname $0)/temp/$mutation
python $(dirname $0)/score_n.py $mutation || exit 1

