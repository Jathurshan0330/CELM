site='S0002'
model_name='meta-llama/Llama-3.2-3B-Instruct' #'meta-llama/Llama-3.1-8B-Instruct'
split='test'
num_workers=8
device=5
experiment_name='test'
# use_accelerate=False

python unimodal_text_and_eeg_features_baseline.py \
    --model_name $model_name \
    --site $site \
    --split $split \
    --split_type random_split_data_by_patient \
    --normalize_eeg_method div_by_100 \
    --task unimodal_text_and_eeg_features_baseline \
    --load_eeg True \
    --num_workers $num_workers \
    --device $device \
    --experiment_name $experiment_name \


python run_evaluate.py \
    --experiment_group unimodal_text_and_eeg_features_baseline \
    --model_name $model_name \
    --experiment_name $experiment_name \
    --site $site \
    --split_type random_split_data_by_patient \
    --split $split \