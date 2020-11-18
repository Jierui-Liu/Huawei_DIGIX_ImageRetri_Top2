
#!/bin/bash
#victor 2020.9.12

mutation=$1

echo $mutation
mkdir -p $(dirname $0)/newest_model_saved/$mutation


/home/LiuJierui/.conda/envs/pytorch/bin/python $(dirname $0)/copy_nofc.py $mutation 0 || exit 1;

