#! /bin/bash


#max_structure=300
max_structure_len=1024
max_cell_len=1024

#max_structure=1536
# Create dataset for train and validation
src_pubtabnet_data_path='pubtabnet'
DATA_FOLDER="output_w_none_stru_${max_structure_len}_cellClip_100"

if [[ -d "${DATA_FOLDER}" ]]; then
    echo "${DATA_FOLDER} is existing"
else
    echo "${DATA_FOLDER} is not existing"
    echo "Create dataset ..."
    python create_input.py \
    --image_folder $src_pubtabnet_data_path \
    --output_folder $DATA_FOLDER \
    --max_len_token_structure ${max_structure_len} \
    --max_len_token_cell ${max_cell_len}
fi

model_dir='checkpoints'
backbone='resnext101_32x8d'
image_size=640
#backbone='resnet18'
#image_size=448

#STAGE='structure'
STAGE='cell'

PYTHON_FILE=train.py

if [[ "$STAGE" == "structure" ]]; then
    hyper_loss=1.0
else
    hyper_loss=0.5
fi
echo "$hyper_loss"


EDD_type='S1S1'
#EDD_type='S2S2'
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CMD="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $PYTHON_FILE \
       --data_folder ${DATA_FOLDER} \
       --num_epochs 26 \
       --batch_size 1 \
       --learning_rate 1e-3 \
       --model_dir $model_dir \
       --backbone $backbone \
       --EDD_type $EDD_type \
       --stage $STAGE \
       --hyper_loss $hyper_loss \
       --first_epoch 1 \
       --second_epoch 1 \
       --print_freq 1 \
       --grad_clip 5.0 \
       --image_size $image_size \
       --max_len_token_structure $max_structure_len "

if [[ ! -z $1  ]]; then
    CMD+="--pretrained_model_path $1 --resume "
fi

$CMD
