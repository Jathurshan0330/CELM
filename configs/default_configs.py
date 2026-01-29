HARVARD_DATASET_CONFIG = {
    'data_root': '[PATH_TO_PROCESSED_EEG_REPORT_DATA]',
    'seed': 5,
    'fs': 200,
}

UNIMODAL_TEXT_ONLY_BASELINE_CONFIG = {
    'save_dir': '[PATH_TO_EXPERIMENT_RESULTS]/unimodal_text_only_baseline',
}

UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG = {
    'save_dir': '[PATH_TO_EXPERIMENT_RESULTS]/unimodal_text_and_eeg_features_baseline',
    'num_seg_to_combine_for_pooling': 10,
    'max_eeg_sequence_length': 1000,
}

EEG_LLM_PROJECTION_ONLY_CONFIG = {
    'save_dir': '[PATH_TO_EXPERIMENT_RESULTS]/eeg_llm_projection_only',
}
