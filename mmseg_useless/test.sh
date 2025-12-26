dir=$1
model=$2
gpu_num=$3
for iternum in $(seq 72000 4000 80000)
do
    bash ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/iter_${iternum}.pth $gpu_num --eval mIoU #--format-only --eval-options imgfile_prefix=./results/${model}/${iternum}
done
# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_8.pth 8 --format-only --eval-options submission_dir=./results/e8_${model}
# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_12.pth 8 --format-only --eval-options submission_dir=./results/e12_${model}
# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_11.pth 8 --format-only --eval-options submission_dir=./results/e11_${model}
