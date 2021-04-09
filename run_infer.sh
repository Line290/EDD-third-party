#!/bin/bash


# Create dataset for validation and test
src_pubtabnet_data_path='pubtabnet'
DATA_FOLDER="output_w_none_all"

if [[ -d "${DATA_FOLDER}" ]]; then
    echo "${DATA_FOLDER} is existing"
else
    echo "${DATA_FOLDER} is not existing"
    echo "Create dataset ..."
    python create_input.py \
    --image_folder $src_pubtabnet_data_path \
    --output_folder $DATA_FOLDER \
    --max_len_token_structure 999999999 \
    --max_len_token_cell 9999999999
fi

MODEL=$1
SPLIT=$2
backbone="resnext101_32x8d"
word_map_structure="${DATA_FOLDER}/WORDMAP_STRUCTURE.json"
word_map_cell="${DATA_FOLDER}/WORDMAP_CELL.json"
beam_size_structure=3
beam_size_cell=3
T=0.65

offset=1150

#for gpu_id in `seq 0 7`
for gpu_id in 0
do
    CUDA_VISIBLE_DEVICES=$((gpu_id)) nohup python inference.py \
    --model $MODEL \
    --data_folder ${DATA_FOLDER} \
    --word_map_structure $word_map_structure \
    --word_map_cell $word_map_cell \
    --beam_size_structure $beam_size_structure \
    --beam_size_cell $beam_size_cell \
    --max_seq_len_structure 1536 \
    --max_seq_len_cell 300 \
    --backbone $backbone \
    --EDD_type "S1S1" \
    --image_size 640 \
    --split $SPLIT \
    --print_freq 100 \
    --all \
    --T $T \
    --rank_method "sum" \
    --start_idx $((gpu_id*$offset)) \
    --offset $offset > log_${SPLIT}_all_part_$((gpu_id*$offset))_$((gpu_id*$offset+$offset)).txt &
done

