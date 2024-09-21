export CUDA_VISIBLE_DEVICES=$1

strategy=$2
tp=None
model=$3
dataset=$4
sample=$5
valid_type=Accuracy
num_cl=1
transition_matrix=uniform

output_dir="/tmp2/yahcreeper/test/libcll/logs/${strategy}_distributed/${dataset}-multi_label_${num_cl}-${transition_matrix}/${strategy}-${tp}-${model}-${dataset}"
output_dir="/tmp2/yahcreeper/test/libcll/logs/test_${strategy}_record/"
output_dir="/tmp2/yahcreeper/test/libcll/logs/test_${strategy}_record_multi/"
output_dir="/tmp2/yahcreeper/test/libcll/logs/test/"

if [[ ${dataset} == "cifar10" ]]; then
    depth=28
    wid=2
    weight_decay=5e-4
elif [[ ${dataset} == "cifar100" ]]; then
    depth=28
    wid=8
    weight_decay=1e-3
fi

python scripts/train.py \
    --do_train \
    --do_predict \
    --strategy ${strategy} \
    --type ${tp} \
    --model ${model} \
    --dataset ${dataset} \
    --lr 3e-2 \
    --batch_size 64 \
    --epoch -1 \
    --augment \
    --valid_type ${valid_type} \
    --output_dir ${output_dir} \
    --num_cl ${num_cl} \
    --transition_matrix ${transition_matrix}\
    --ssl \
    --valid_split 0 \
    --samples_per_class ${sample} \
    --max_steps 1048576 \
    --depth ${depth} \
    --widen_factor ${wid} \
    --weight_decay ${weight_decay} \
