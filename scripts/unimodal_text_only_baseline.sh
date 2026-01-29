site='S0002'
model_name='Qwen/Qwen3-4B-Instruct-2507'
split='test'
num_workers=8
device=5
experiment_name='test'
# use_accelerate=False

python unimodal_text_only_baseline.py \
    --model_name $model_name \
    --site $site \
    --split $split \
    --split_type random_split_data_by_patient \
    --normalize_eeg_method div_by_100 \
    --task unimodal_text_only_baseline \
    --load_eeg False \
    --num_workers $num_workers \
    --device $device \
    --experiment_name $experiment_name \


python run_evaluate.py \
    --experiment_group unimodal_text_only_baseline \
    --model_name $model_name \
    --experiment_name $experiment_name \
    --site $site \
    --split_type random_split_data_by_patient \
    --split $split \