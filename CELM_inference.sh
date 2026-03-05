
# Model parameters
eeg_encoder_model='cbramod'
llm_model='Qwen/Qwen3-4B-Instruct-2507'
projector='sequence_transformer_linear_projector' # 'sequence_transformer_perceiver_projector'  
training_mode='projection_only'

# Dataset parameters
site='S0002'
split_type='random_split_data_by_patient'
normalize_eeg_method='div_by_100'
task='eeg_llm_projection_only'
num_workers=8


experiment_name='CELM_SCA_projector_inference'
device=5


# Adjust the projector checkpoint path
llm_model_name='Qwen3-4B-Instruct-2507'
projector_checkpoint_path='./pretrained_CELM_projectors/SCA_cbramod_Qwen3-4B-Instruct-2507_S0002/checkpoint_epoch_9.pt'
results_saved_path="./results/eeg_llm_projection_only/${experiment_name}/${eeg_encoder_model}_${llm_model_name}/inference_results_${site}/checkpoint_epoch_9"

python CELM_inference.py \
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
    --checkpoint_path $projector_checkpoint_path \


python run_evaluate_CELM.py \
    --model_name $llm_model \
    --results_saved_path $results_saved_path \
    --site $site \
    --split_type $split_type \
    --split test \
    --ignore_perplexity 