HARVARD_DATASET_CONFIG = {
    'data_root': '/shared/rsaas/jp65/harvard_eeg_data/processed_harvard_EEG_and_report_data',
    'seed': 5,
    'fs': 200,
}

UNIMODAL_TEXT_ONLY_BASELINE_CONFIG = {
    # 'save_dir': '/shared/rsaas/jp65/harvard_eeg_data/EEG_Report_Generative_FM/experiment_results/unimodal_text_only_baseline',
    'save_dir': '/home/jp65/EEG_Report_Generative_FM/experiment_results/unimodal_text_only_baseline',
}

UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG = {
    # 'save_dir': '/shared/rsaas/jp65/harvard_eeg_data/EEG_Report_Generative_FM/experiment_results/unimodal_text_and_eeg_features_baseline',
    'save_dir': '/home/jp65/EEG_Report_Generative_FM/experiment_results/unimodal_text_and_eeg_features_baseline',
    'num_seg_to_combine_for_pooling': 10,
    'max_eeg_sequence_length': 1000,
}

EEG_LLM_PROJECTION_ONLY_CONFIG = {
    # 'save_dir': '/shared/rsaas/jp65/harvard_eeg_data/EEG_Report_Generative_FM/experiment_results/eeg_llm_projection_only',
    'save_dir': '/home/jp65/EEG_Report_Generative_FM/experiment_results/eeg_llm_projection_only',
}



TUEV_DATASET_CONFIG = {
    'data_root': '/srv/local/data/jp65/TUEV_processed_256/' ,
    'seed': 5,
    'fs': 256,
    'num_classes': 6,
    'num_channels': 16,
}