dir=$1
model=$2
gpu_num=$3
bash ./tools/dist_train.sh ./configs/${dir}/${model}.py $gpu_num
