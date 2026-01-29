source /home/jp65/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate LSE_EEG
cd /home/jp65/EEG_Report_Generative_FM/EEG_Report_Generative_FM/data_preprocessing




# Extract and format neurology reports using Meta-Llama-3-8B-Instruct
site='[SITE_ID]' # S0001 or S0002
data_path='[PATH_TO_NEUROLOGY_REPORTS]' # I0001_neurology_reports
HEEDB_patients_path='[PATH_TO_HEEDB_METADATA]' # HEEDB_Metadata
save_path='[PATH_TO_SAVE_PROCESSED_NEUROLOGY_REPORTS]'
notes_path='[PATH_TO_SAVED_PROCESSED_NEUROLOGY_REPORTS]'
site_id_filter='S0001'
model_name='meta-llama/Meta-Llama-3-8B-Instruct' 
device=0
num_repetitions=5
overwrite_existing_reports=False
start_index=0
# end_index=2000


python extract_and_format_neurology_reports_1.py  --data_path $data_path \
                                                --save_path $save_path \
                                                --notes_path $notes_path \
                                                --site_id_filter $site_id_filter \
                                                --HEEDB_patients_path $HEEDB_patients_path \
                                                --model_name $model_name \
                                                --device $device \
                                                --num_repetitions $num_repetitions \
                                                --overwrite_existing_reports $overwrite_existing_reports \
                                                --start_index $start_index \
                                                # --end_index $end_index


# Match reports with recordings and download the recordings directly from AWS S3 # TODO: Update the paths in the script
python match_reports_with_recordings_2.py --site $site


# Preprocess the recordings # TODO: Update the paths in the script
python preprocess_eeg_3.py --site $site

# Create description_df # TODO: Update the paths in the script
python create_description_df_4_eff.py --site $site --num_workers 8

