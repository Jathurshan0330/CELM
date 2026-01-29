import os
import json
import pickle
import glob
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
from utils.utils import seed_everything, bandpower_segments

from configs.default_configs import HARVARD_DATASET_CONFIG, UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG
from configs.section_mapping import SECTION_STANDARDIZATION_MAPPING, STANDARDIZED_SECTION_DESCRIPTIONS
from .prompts import PromptGenerator


@dataclass
class HarvardEEGReportSample:
    "Single EEG-Report Sample"
    eeg_segments: List[torch.Tensor] # list of torch tensors where each tensor is a EEG session with shape of segments x channels x times 
    available_channels: List[List[str]] # Available channels in each recordings
    eeg_report: Dict        # Dict of eeg sections in the report
    clinical_history: str   # Combined string of clinical history sections in the report
    meta_data: Dict         # meta_data such as Age, Gender
    generated_prompt: str             # prompt for the report
    labels: str                      # labels for the report
    


class HarvardEEGReportDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        site: str,
        report_eeg_sections: List[str] = None,
        split: str = "train",
        split_type: str = 'random_split_data_by_patient',
        normalize_eeg_method: str = "div_by_100", # 'z-score_norm', 'div_by_100', 'div_by_95_quartile'
        task: str = 'unimodal_text_only_baseline', # 'unimodal_text_only_baseline', 'unimodal_text_and_eeg_features_baseline', 'eeg_llm_projection_only','abnormal_prediction', 'unimodal_text_and_eeg_features_abnormal_prediction'
        load_eeg: bool = True, # if True, load the eeg data
        combine_k: int = None, # if None, do not combine the eeg data
        drop_last: bool = False, # if True, drop the last segment if it is not a multiple of combine_k
        max_eeg_sequence_length: int = None, # if None, do not truncate the eeg data
        ):
        
        self.matched_eeg_report_path = os.path.join(data_root,'matched_eeg_recordings_report',site)
        self.split_csv = pd.read_csv(os.path.join(data_root,split_type,f'{site}_{split}_split.csv'))
        self.normalize_eeg_method = normalize_eeg_method
        self.prompt_generator = PromptGenerator(task=task)
        self.task = task
        self.load_eeg = load_eeg
        self.combine_k = combine_k
        self.drop_last = drop_last
        self.max_eeg_sequence_length = max_eeg_sequence_length
        if report_eeg_sections!= None:
            self.report_eeg_sections = report_eeg_sections
        else:
            self.report_eeg_sections = 'all'
        self.eeg_channels = [
                    'C3', 'C4', 'O1', 'O2', 'Cz',
                    'F3', 'F4', 'F7', 'F8', 'Fz',
                    'Fp1', 'Fp2', 'Fpz',
                    'P3', 'P4', 'Pz',
                    'T3', 'T4', 'T5', 'T6',
                    'A1', 'A2'
                ]
            
        
        print('Number of Report-EEG Samples', len(self.split_csv))
        
        
    def __len__(self):
        return len(self.split_csv)
    

    def read_eeg(self,processed_eeg_path_list: List[str]):
        eeg = []
        available_channels_list = []
        for processed_eeg_path in processed_eeg_path_list:
            eeg_files_list = glob.glob(f'{processed_eeg_path}/*.pkl')
            eeg_file_temp = []
            available_channels_temp=None
            for eeg_file in eeg_files_list:
                signal_data = pickle.load(open(eeg_file, 'rb'))
                x = signal_data['signal']
                available_channels_temp = signal_data['available_channels']                
                if self.normalize_eeg_method == 'z-score_norm':
                    x = (x-signal_data['mean'])/(signal_data['std']+1e-12)
                elif self.normalize_eeg_method == "div_by_100":
                    x = x/100
                elif self.normalize_eeg_method == 'div_by_95_quartile':
                    x = x/(np.quantile(np.abs(x), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-12)
                
                eeg_file_temp.append(np.array(x))
                
            available_channels_list.append(available_channels_temp)
            eeg_file_temp = np.array(eeg_file_temp)
            if self.max_eeg_sequence_length is not None:
                eeg_file_temp = eeg_file_temp[:self.max_eeg_sequence_length]
            eeg_file_temp = torch.FloatTensor(eeg_file_temp)
            eeg.append(eeg_file_temp)
        
        return eeg, available_channels_list
    
    def read_eeg_and_combine(self, processed_eeg_path_list: List[str], combine_k: int = 3, drop_last: bool = False):
        eeg = []
        available_channels_list = []

        for processed_eeg_path in processed_eeg_path_list:
            eeg_files_list = sorted(glob.glob(f"{processed_eeg_path}/*.pkl"))  # sorted for consistent order
            eeg_file_temp = []
            available_channels_temp = None

            for eeg_file in eeg_files_list:
                signal_data = pickle.load(open(eeg_file, "rb"))
                x = signal_data["signal"]                      # (C, T) usually
                available_channels_temp = signal_data["available_channels"]

                if self.normalize_eeg_method == "z-score_norm":
                    x = (x - signal_data["mean"]) / (signal_data["std"] + 1e-12)
                elif self.normalize_eeg_method == "div_by_100":
                    x = x / 100
                elif self.normalize_eeg_method == "div_by_95_quartile":
                    x = x / (np.quantile(np.abs(x), q=0.95, axis=-1, method="linear", keepdims=True) + 1e-12)

                eeg_file_temp.append(np.asarray(x, dtype=np.float32))

            available_channels_list.append(available_channels_temp)

            # [N, C, T]
            eeg_file_temp = np.stack(eeg_file_temp, axis=0)

            # --- combine every k segments into one by concatenating time ---
            N, C, T = eeg_file_temp.shape
            n_groups = N // combine_k if drop_last else int(np.ceil(N / combine_k))

            combined = []
            for g in range(n_groups):
                start = g * combine_k
                end = min((g + 1) * combine_k, N)
                chunk = eeg_file_temp[start:end]  # [k', C, T]

                if chunk.shape[0] < combine_k:
                    if drop_last:
                        break
                    # pad by repeating last segment to reach k
                    pad = np.repeat(chunk[-1:, :, :], combine_k - chunk.shape[0], axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)

                # concat along time: [k, C, T] -> [C, k*T]
                chunk_cat = np.concatenate([chunk[i] for i in range(combine_k)], axis=-1)
                combined.append(chunk_cat)

            # [N_new, C, k*T]
            eeg_tensor = torch.from_numpy(np.stack(combined, axis=0)) if len(combined) > 0 else torch.empty((0, C, combine_k*T))
            eeg.append(eeg_tensor)

        return eeg, available_channels_list
    
    def get_eeg_statistics(self,eeg):
        eeg_statistics = {}
        for ind in range(len(eeg)):
            if eeg[ind].shape[0] > self.max_eeg_sequence_length:
                print(f'eeg[ind].shape[0] > self.max_eeg_sequence_length: {eeg[ind].shape[0]} > {self.max_eeg_sequence_length}')
                eeg[ind] = eeg[ind][:self.max_eeg_sequence_length]
            band_power = bandpower_segments(np.array(eeg[ind]),fs=HARVARD_DATASET_CONFIG['fs'],
                                            num_seg_to_combine_for_pooling=UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG['num_seg_to_combine_for_pooling'],
                                            bands=None,nperseg=None,noverlap=None,
                                            window="hann",detrend="constant",scaling="density",
                                            relative=True,eps=1e-12)
            # band_power = np.round(band_power,2)
            eeg_stat_temp = {'delta (0.5-4Hz) band power (dB)':np.vectorize(lambda x: f"{x:.2f}")(band_power[:,:,0]).tolist(),
                             'theta (4-8Hz) band power (dB)':np.vectorize(lambda x: f"{x:.2f}")(band_power[:,:,1]).tolist(),
                             'alpha (8-12Hz) band power (dB)':np.vectorize(lambda x: f"{x:.2f}")(band_power[:,:,2]).tolist(),
                             'beta (12-30Hz) band power (dB)':np.vectorize(lambda x: f"{x:.2f}")(band_power[:,:,3]).tolist(),
                             'gamma (30-80Hz) band power (dB)':np.vectorize(lambda x: f"{x:.2f}")(band_power[:,:,4]).tolist()}
            eeg_statistics[f'eeg_session_{ind}'] = eeg_stat_temp
        return eeg_statistics
    
    def read_report(self,report_deidentified_name,meta_data):
        report_path = os.path.join(self.matched_eeg_report_path,report_deidentified_name,f'{report_deidentified_name}.json')
        # read json
        report_json = json.load(open(report_path))
        eeg_report_dict = {"EEG_sections":[],
                           "extracted_eeg_section_names":[]}
        
        if self.report_eeg_sections == 'all':
            eeg_report_dict["EEG_sections"] = report_json["EEG_section_llm_extractions"]["EEG_sections"]
            eeg_report_dict["extracted_eeg_section_names"] = report_json["extracted_eeg_section_names"]
        else:
            for dict_ in report_json["EEG_section_llm_extractions"]["EEG_sections"]:
                if dict_['section_name'] in self.report_eeg_sections:
                    eeg_report_dict["EEG_sections"].append(dict_)
                    eeg_report_dict["extracted_eeg_section_names"].append(dict_['section_name'])
                    
        # Standardize the eeg section names
        eeg_report_dict["extracted_eeg_section_names"] = [SECTION_STANDARDIZATION_MAPPING[section_name] for section_name in eeg_report_dict["extracted_eeg_section_names"]]
        eeg_report_dict["EEG_sections"] = [{"section_name":SECTION_STANDARDIZATION_MAPPING[dict_['section_name']], "section_text":dict_['section_text']} for dict_ in eeg_report_dict["EEG_sections"] ]
        
                    
        clinical_history = f'age: {meta_data["Avg_Age"]}\ngender: {meta_data["Gender"]}\n'
        if 'patient_history_section_llm_extractions' in report_json.keys():
            for dict_ in report_json["patient_history_section_llm_extractions"]["CLINICAL_sections"]:
                section_name = dict_['section_name']
                section_text = dict_['section_text']
                clinical_history+=f'{section_name}\n{section_text}\n\n'
            
        return eeg_report_dict, clinical_history
    
    def get_labels(self,eeg_report_dict):
        labels = {'report_sections':eeg_report_dict['EEG_sections']}
        # convert to json string
        labels = json.dumps(labels, ensure_ascii=False)
        return labels
        
    
    def create_text_prompt(self, 
                           report_eeg_sections: List[str] = None, 
                           clinical_history: str = None,
                           eeg_statistics: str = None,
                           eeg_channels: List[str] = None):
        gen_prompt = self.prompt_generator.get_prompt()
        if self.task == 'unimodal_text_only_baseline':
            gen_prompt = gen_prompt.replace('[PATIENT_HISTORY_AND_EEG_DESCRIPTION]', clinical_history)
        elif self.task == 'unimodal_text_and_eeg_features_baseline':
            gen_prompt = gen_prompt.replace('[PATIENT_HISTORY_AND_EEG_DESCRIPTION]', clinical_history)
            gen_prompt = gen_prompt.replace('[EEG_DERIVED_STATISTICS]', f'{eeg_statistics}')
            gen_prompt = gen_prompt.replace('[EEG_CHANNELS]', f'{eeg_channels}')
        elif self.task == 'eeg_llm_projection_only':
            gen_prompt = gen_prompt.replace('[PATIENT_HISTORY_AND_EEG_DESCRIPTION]', clinical_history)
            gen_prompt = gen_prompt.replace('[EEG_CHANNELS]', f'{eeg_channels}')
        elif self.task == 'abnormal_prediction':
            gen_prompt = gen_prompt.replace('[PATIENT_HISTORY_AND_EEG_DESCRIPTION]', clinical_history)
            gen_prompt = gen_prompt.replace('[EEG_CHANNELS]', f'{eeg_channels}')
            return gen_prompt
        elif self.task == 'unimodal_text_and_eeg_features_abnormal_prediction':
            gen_prompt = gen_prompt.replace('[PATIENT_HISTORY_AND_EEG_DESCRIPTION]', clinical_history)
            gen_prompt = gen_prompt.replace('[EEG_DERIVED_STATISTICS]', f'{eeg_statistics}')
            gen_prompt = gen_prompt.replace('[EEG_CHANNELS]', f'{eeg_channels}')
            return gen_prompt
        gen_prompt = gen_prompt.replace('[SECTION_NAMES]', f'{report_eeg_sections}')
        standardized_section_descriptions = '\n'.join([f'{section_name}: {STANDARDIZED_SECTION_DESCRIPTIONS[section_name]}' for section_name in report_eeg_sections])
        gen_prompt = gen_prompt.replace('[STANDARDIZED_SECTION_DESCRIPTIONS]', f'{standardized_section_descriptions}')
        return gen_prompt
    
    def __getitem__(self, idx): 
        idx_row = self.split_csv.iloc[idx]
        report_deidentified_name = idx_row['DeidentifiedName(Reports)']
        report_deidentified_name = report_deidentified_name.replace('.txt','')
        
        meta_data = idx_row.to_dict()
        
        processed_eeg_path_list = str(idx_row['Processed_EEG_Paths']).split(',')
        processed_eeg_path_list = [os.path.join(self.matched_eeg_report_path,report_deidentified_name,x) for x in processed_eeg_path_list]
        
        if self.load_eeg:
            if self.combine_k is not None:
                eeg, available_channels = self.read_eeg_and_combine(processed_eeg_path_list, self.combine_k, self.drop_last)
            else:
                eeg, available_channels = self.read_eeg(processed_eeg_path_list)
        else:
            eeg = [None]*len(processed_eeg_path_list)
            available_channels = [None]*len(processed_eeg_path_list)
        
        
        eeg_report_dict, clinical_history = self.read_report(report_deidentified_name, meta_data)
        
        
        
        if self.task == 'unimodal_text_only_baseline':
            generated_prompt = self.create_text_prompt(report_eeg_sections=eeg_report_dict['extracted_eeg_section_names'], 
                                                       clinical_history=clinical_history)
            labels = None
        elif self.task == 'unimodal_text_and_eeg_features_baseline':
            eeg_statistics = self.get_eeg_statistics(eeg)
            eeg_statistics = json.dumps(eeg_statistics, ensure_ascii=False)
            generated_prompt = self.create_text_prompt(report_eeg_sections=eeg_report_dict['extracted_eeg_section_names'], clinical_history=clinical_history, eeg_statistics=eeg_statistics, eeg_channels=self.eeg_channels)
            labels = None
        elif self.task == 'eeg_llm_projection_only':
            generated_prompt = self.create_text_prompt(report_eeg_sections=eeg_report_dict['extracted_eeg_section_names'], clinical_history=clinical_history, eeg_channels=self.eeg_channels)
            labels = self.get_labels(eeg_report_dict)
        elif self.task == 'abnormal_prediction':
            generated_prompt = self.create_text_prompt(report_eeg_sections=eeg_report_dict['extracted_eeg_section_names'], clinical_history=clinical_history, eeg_channels=self.eeg_channels)
            labels = None
        elif self.task == 'unimodal_text_and_eeg_features_abnormal_prediction':
            eeg_statistics = self.get_eeg_statistics(eeg)
            eeg_statistics = json.dumps(eeg_statistics, ensure_ascii=False)
            generated_prompt = self.create_text_prompt(report_eeg_sections=eeg_report_dict['extracted_eeg_section_names'], clinical_history=clinical_history, eeg_statistics=eeg_statistics, eeg_channels=self.eeg_channels)
            labels = None
        return HarvardEEGReportSample(
            eeg_segments=eeg,
            available_channels=available_channels,
            eeg_report=eeg_report_dict,
            clinical_history=clinical_history,
            meta_data=meta_data,
            generated_prompt=generated_prompt,
            labels=labels
        )
        
        
        
    
def harvard_eeg_report_collate_fn(batch: List[HarvardEEGReportSample]) -> Dict[str, object]:
    return {
        "eeg_segments": [b.eeg_segments for b in batch],                # list (batch) of list (recordings) of tensors
        "available_channels": [b.available_channels for b in batch],    # list (batch) of list (recordings) of channel lists
        "eeg_report": [b.eeg_report for b in batch],                    # list of dicts
        "clinical_history": [b.clinical_history for b in batch],        # list of strings
        "meta_data": [b.meta_data for b in batch],                      # list of dicts
        "generated_prompt": [b.generated_prompt for b in batch],        # list of strings
        "labels": [b.labels for b in batch],                          # list of strings
    }
    
    
def get_harvard_data_loader(
    site: str,
    report_eeg_sections: List[str] = None,
    split: str = "train",
    split_type: str = 'random_split_data_by_patient',
    normalize_eeg_method: str = "div_by_100", # 'z-score_norm', 'div_by_100', 'div_by_95_quartile'
    task: str = 'unimodal_text_only_baseline', # 'unimodal_text_only_baseline', 'unimodal_text_and_eeg_features_baseline', 'eeg_llm_projection_only', 'abnormal_prediction', 'unimodal_text_and_eeg_features_abnormal_prediction'
    load_eeg: bool = True, # if True, load the eeg data
    batch_size = 32, 
    num_workers=8,
    combine_k: int = None,
    drop_last: bool = False,
    max_eeg_sequence_length: int = None,
    ):
    
    
    data_root = HARVARD_DATASET_CONFIG['data_root']
    seed = HARVARD_DATASET_CONFIG['seed']
    seed_everything(seed)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
        
    harvard_eeg_dataset = HarvardEEGReportDataset(data_root=data_root,
                                        site = site,
                                        report_eeg_sections = report_eeg_sections,
                                        split = split,
                                        split_type = split_type,
                                        normalize_eeg_method = normalize_eeg_method,
                                        task = task,
                                        load_eeg = load_eeg,
                                        combine_k = combine_k,
                                        drop_last = drop_last,
                                        max_eeg_sequence_length = max_eeg_sequence_length,
                                        )
    
    harvard_eeg_loader = torch.utils.data.DataLoader(harvard_eeg_dataset, 
                                                batch_size = batch_size, 
                                                shuffle = shuffle, 
                                                num_workers = num_workers,
                                                collate_fn = harvard_eeg_report_collate_fn)
    
    return harvard_eeg_loader