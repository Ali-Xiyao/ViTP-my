dir=$1
model=$2
gpu_num=$3
bash train.sh $dir $model $gpu_num 
bash test.sh $dir $model $gpu_num