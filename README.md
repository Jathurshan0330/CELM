# Clinical EEG Language Model (CLEM): Neural Signals Generate Clinical Notes in the Wild

[![arXiv](https://img.shields.io/badge/arXiv-2601.22197-b31b1b.svg)](https://arxiv.org/abs/2601.22197)
[![Slides](https://img.shields.io/badge/Slides-PDF-F5A623.svg)](https://drive.google.com/file/d/13Nb880p0EKJfzANlibfvUjC96GMdsvHG/view?usp=sharing)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Jathurshan/CELM)
[![Workshop](https://img.shields.io/badge/ICLR_2026-Workshop_on_MultiModal_Intelligence-4b44ce.svg)](https://openreview.net/forum?id=OgSSKlnfBQ)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Jathurshan0330/CELM?style=social)](https://github.com/Jathurshan0330/CELM)


<img width="1182" height="1000" alt="43794a7a-1" src="https://github.com/user-attachments/assets/b7502104-a7df-4ad9-9bef-4cfd727558aa" />

## 💡 Overview

Generating clinical reports that summarize abnormal patterns, diagnostic findings, and clinical interpretations from long-term EEG recordings remains labor-intensive. We curate a largescale clinical EEG dataset with 9,922 reports paired with approximately 11,000 hours of EEG recordings from 9,048 patients. 

We therefore develop CELM, the first clinical EEG-to-Language foundation model capable of summarizing longduration, variable-length EEG recordings and performing end-to-end clinical report generation at multiple scales, including recording description,
background activity, epileptiform abnormalities, events/seizures, and impressions. Experimental results show that, with patient history supervision, our method achieves 70%–95% average relative improvements in standard generation metrics (e.g., ROUGE-1 and METEOR) from 0.2–0.3to 0.4–0.6. In the zero-shot setting without patient history, CELM attains generation scores in the range of 0.43–0.52, compared to baselines of 0.17–0.26. CELM integrates pretrained EEG foundation models with language models to enable scalable multimodal learning. We release our model and benchmark construction pipeline in this repository.




## 📰 News
- **[Work in Progress]** releasing model on HuggingFace.
- **[2026-03-05]** Code released on [GitHub]()!
- **[2026-03-02]** Accepted to ICLR 2026 Workshop MM Intelligence!
- **[2026-01-29]** Preprint released on [arXiv](https://arxiv.org/abs/2601.22197)!

## 🥇 Key Results

| Setting | Metric | Best Baseline | CELM |
|---|---|---|---|
| With patient history - S0001 | ROUGE-1 | 0.3038 | **0.5565** (83% relative improvement) |
| With patient history - S0002 | ROUGE-1 | 0.2886 | **0.6408** (122% relative improvement) |
| Zero-context (EEG only input) - S0002 | ROUGE-1 | 0.2557 | **0.5248** (105% relative improvement) |

CELM generates clinical reports at multiple scales: recording description, background activity, epileptiform abnormalities, events/seizures, and impressions.

## 🚀 Quickstart

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

## CELM Inference
Run the following scripts using the pretrained weights for the projectors in ./pretrained_CELM_projectors, and infer our CELM. Make sure to download the checkpoints for CBraMod from  https://huggingface.co/weighting666/CBraMod and add them to ./eeg_encoders/pretrained_checkpoints.
```
./CELM_inference.sh
```

## CELM Training and Inference
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

## 📝 Citation
If you find this work useful, please consider citing our paper and giving this repo a ⭐!
```
@article{pradeepkumar2026neural,
  title={Neural Signals Generate Clinical Notes in the Wild},
  author={Pradeepkumar, Jathurshan and Chen, Zheng and Sun, Jimeng},
  journal={arXiv preprint arXiv:2601.22197},
  year={2026}
}
```

## Acknowledgements

We thank the [Harvard EEG Database](https://bdsp.io/content/harvard-eeg-db/4.1/) and the [CBraMod](https://huggingface.co/weighting666/CBraMod) teams for making their data and models publicly available.

We appreciate your interest in our work! 😃😃😃😃😃

<div align="center">
  <sub>Built with ❤️ for the clinical neuroscience community</sub>
</div>
