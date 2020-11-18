
#!/bin/bash
#victor 2020.9.12

mutation=$1

echo $mutation
mkdir -p $(dirname $0)/embeddings/$mutation/nonlocal
mkdir -p $(dirname $0)/embeddings/$mutation/rag
mkdir -p $(dirname $0)/embeddings/$mutation/attention


# /home/LiuJierui/.conda/envs/pytorch/bin/python $(dirname $0)/copy_nofc.py $mutation 0 || exit 1;
python $(dirname $0)/extract_multi.py $mutation || exit 1;


# mkdir -p $(dirname $0)/submission_file
# mkdir -p $(dirname $0)/temp/$mutation
# python $(dirname $0)/score_n_nonlocal.py $mutation || exit 1
# python $(dirname $0)/score_n_rag.py $mutation || exit 1
# python $(dirname $0)/score_n_attention.py $mutation || exit 1

