
#!/bin/bash
#victor 2020.9.12

mutation=$1
ifresume_training=$2

if [ $ifresume_training == false ] 
then

mkdir -p $(dirname $0)/log
mkdir -p $(dirname $0)/best_model_saved/$mutation
mkdir -p $(dirname $0)/newest_model_saved/$mutation


rm -rf $(dirname $0)"/log/"$mutation   #删除上次训练的记录


python $(dirname $0)/train.py $mutation 0 || exit 1;


else

echo "resume training"
python $(dirname $0)/train.py $mutation 1 || exit 1;

fi




# #extraction

# mkdir -p $(dirname $0)/embeddings/$mutation
# /home/LiuJierui/.conda/envs/pytorch/bin/python $(dirname $0)/extract.py $mutation || exit 1;



# mkdir -p $(dirname $0)/submission_file
# mkdir -p $(dirname $0)/temp/$mutation
# /home/LiuJierui/.conda/envs/pytorch/bin/python $(dirname $0)/score.py $mutation;

# /bin/rm -rf $(dirname $0)/temp/$mutation

