#!/bin/bash
#victor 2019.1.12


numofgpurequired=$1


gpu_status=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader);
gpu_mapping=$(nvidia-smi --query-gpu=gpu_uuid,index --format=csv,noheader);

# gpu_status="GPU-5b3fe3d3-c237-7c44-44b5-fa30543eb587, 7379 GPU-1806a51a-9c18-ffce-816b-6528617d451d, 7379 GPU-9afd5e20-4260-b8e8-ea7f-54bca05b8f5b, 7379"
# #gpu_status="GPU-5b3fe3d3-c237-7c44-44b5-fa30543eb587, 0 GPU-7c038201-2b4c-2f0c-b908-a55220352fbb, 1 GPU-1806a51a-9c18-ffce-816b-6528617d451d, 2 GPU-9afd5e20-4260-b8e8-ea7f-54bca05b8f5b, 3 GPU-95d4d0e6-94c0-17d6-26fb-27e66c6643d6, 4 GPU-5be51993-fc28-2056-8178-f45a055965fc, 5 GPU-0df93b64-d43a-75ee-4002-0540e6669c6a, 6 GPU-cce79931-7088-e2b5-1172-0eb70f2a7397, 7 GPU-499c566e-436e-9bc7-e57b-ae62549b4372, 8 GPU-88527143-893d-abaa-7e45-3dc6943da7a8, 9"
# gpu_mapping="GPU-5b3fe3d3-c237-7c44-44b5-fa30543eb587, 0 GPU-7c038201-2b4c-2f0c-b908-a55220352fbb, 1 GPU-1806a51a-9c18-ffce-816b-6528617d451d, 2 GPU-9afd5e20-4260-b8e8-ea7f-54bca05b8f5b, 3 GPU-95d4d0e6-94c0-17d6-26fb-27e66c6643d6, 4 GPU-5be51993-fc28-2056-8178-f45a055965fc, 5 GPU-0df93b64-d43a-75ee-4002-0540e6669c6a, 6 GPU-cce79931-7088-e2b5-1172-0eb70f2a7397, 7 GPU-499c566e-436e-9bc7-e57b-ae62549b4372, 8 GPU-88527143-893d-abaa-7e45-3dc6943da7a8, 9"
#echo $gpu_status
#echo $gpu_mapping




numofava=0

for ele in $gpu_mapping
do
    if [[ $ele =~ "GPU" ]]
    then
        if [[ $gpu_status =~ $ele ]]
        then
            ava=False

        else
            ava=True
            ((numofava++))

        fi
    else
        if [ x$ava == x'True' ]
        then
        avagpu[$[$numofava-1]]=$ele

        fi
    fi

done


if [ $numofgpurequired -gt $numofava ]
then
echo "available gpu is not enough,using only " $numofava "gpus"
exit 1
fi




counter=0
st=$[RANDOM%$numofava]
unset temp


for((igpu=0;igpu<$numofava;igpu++));  
do   
gpunum=$[$igpu+$st]

gpunum=$[$gpunum%$numofava]

temp=$temp"${avagpu[$gpunum]}"
((counter++))
    if [ $counter -eq $numofgpurequired ]    
    then
        break
    else
    temp=$temp","
    fi
      



done  

echo "using "$temp
export CUDA_VISIBLE_DEVICES=$temp
