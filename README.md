# Clinical EEG Language Model (CLEM): Neural Signals Generate Clinical Notes in the Wild

## Getting Started

Clone the repository:
```
git clone https://github.com/Jathurshan0330/CELM.git
cd CELM
```
Set up environment:
```
conda env create -f setup.yml
conda activate CELM
```

## EEG-Report Benchmark Generation
Our benchmark is built utilizing the dataset from https://bdsp.io/content/harvard-eeg-db/4.1/, which is publicly accessible. Instructions for obtaining access are available on their website. The dataset is large, so we provide the pipeline to download only the necessary files to generate this EEG-Report Benchmark

Run the following script in ./eeg_report_data_construction to construct the EEG-report benchmark efficiently. Make sure to update the data paths and site ID.
```
./eeg_report_data_construction/prepare_eeg_report_benchmark.sh
```
Jupyter notebooks in ./eeg_report_data_construction/data_splits can be used to create the data splits for S0001 and S0002. 


## CELM Training
Run the following script to train the Clinical EEG Language Model. Make sure to download the checkpoints for CBraMod from  https://huggingface.co/weighting666/CBraMod and add them to ./eeg_encoders/pretrained_checkpoints. The script also enables inference and evaluation on the test set.
```
./scripts/CELM_training.sh
```

## Unimodal Baselines

We also provide the scripts to reproduce the unimodal baselines reported in our manuscript. Simply run the following scripts to reproduce the results. Make sure to set the LLM base model name, which loads from HuggingFace 🤗

For Unimodal + Text only baselines
```
./scripts/unimodal_text_only_baseline.sh
```
For Unimodal + Text + EEG Features baselines
```
./scripts/unimodal_text_and_eeg_features_baseline.sh
```

## Citation
If you find our work or this repository interesting and useful, please consider giving a star ⭐.
```

```

We appreciate your interest in our work! 😃😃😃😃😃
