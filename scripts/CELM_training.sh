#!/bin/bash

# Model parameters
eeg_encoder_model='cbramod'
llm_model='google/gemma-3-1b-it'
projector='sequence_transformer_projector' # SCA: sequence_transformer_projector SCC: sequence_transformer_perceiver_projector Linear: linear Perceiver: perceiver_projector
training_mode='projection_only'

# Dataset parameters
site='[SITE_ID]' # S0001 or S0002
split_type='[SPLIT_TYPE]' 
normalize_eeg_method='div_by_100'
task='eeg_llm_projection_only'
num_workers=8

# experiment parameters
experiment_name='[EXPERIMENT_NAME]'
device=0



accelerate launch --gpu_ids="0,1" --num_processes=2 --multi_gpu CELM_training.py \
    --eeg_encoder_model $eeg_encoder_model \
    --llm_model $llm_model \
    --projector $projector \
    --training_mode $training_mode \
    --site $site \
    --split_type $split_type \
    --normalize_eeg_method $normalize_eeg_method \
    --task $task \
    --num_workers $num_workers \
    --device $device \
    --experiment_name $experiment_name \
    --use_accelerate \
    # --resume_from_last_checkpoint
    


lm_model_name='gemma-3-1b-it'
checkpoint_path='[PATH_TO_CHECKPOINT .pt]'
results_saved_path="[PATH_TO_GENERATION_RESULTS]/inference_results_${site}"

python CLEM_inference.py \
    --eeg_encoder_model $eeg_encoder_model \
    --llm_model $llm_model \
    --projector $projector \
    --training_mode $training_mode \
    --site $site \
    --split_type $split_type \
    --normalize_eeg_method $normalize_eeg_method \
    --task $task \
    --num_workers $num_workers \
    --device $device \
    --experiment_name $experiment_name \
    --checkpoint_path $checkpoint_path \


python run_evaluate_CLEM.py \
    --model_name $llm_model \
    --results_saved_path $results_saved_path \
    --site $site \
    --split_type $split_type \
    --split test \
    --ignore_perplexity 




