# import warnings
# warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
from datetime import datetime
import tqdm
import torch
from configs.default_configs import  UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG
from dataset.eeg_report_data_loader import get_harvard_data_loader

from transformers import pipeline
from accelerate import Accelerator

from utils.utils import extract_json, clean_generation_for_json_parsing, seed_everything

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset Parameters
    parser.add_argument('--site', type=str, default='S0002')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--split_type', type=str, default='random_split_data_by_patient')
    parser.add_argument('--normalize_eeg_method', type=str, default='div_by_100')
    parser.add_argument('--task', type=str, default='unimodal_text_and_eeg_features_baseline')
    parser.add_argument('--load_eeg', type=bool, default=True)
    # parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Model Parameters
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    
    # experiment parameters
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_accelerate', action='store_true')
    
    args = parser.parse_args()
    
    if args.use_accelerate:
        args.device = Accelerator().device
    
    # seed everything
    seed_everything(5)
    
    # Create Save Directory
    model_save_name = args.model_name.split('/')[-1]
    if args.experiment_name:
        save_dir = os.path.join(UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG['save_dir'], args.experiment_name, model_save_name)
    else:
        save_dir = os.path.join(UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG['save_dir'], model_save_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_reports_json'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_reports_txt'), exist_ok=True)
    
    # save args
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))
    
    # Create Log File
    log_file = os.path.join(save_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    log_f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    print(f"Log file: {log_file}")
    
    error_generated_reports = {'error_generated_reports': []}
    if os.path.exists(os.path.join(save_dir, 'error_generated_reports.json')):
        with open(os.path.join(save_dir, 'error_generated_reports.json'), 'r') as f:
            error_generated_reports = json.load(f)
    
    # Load Data
    test_data_loader = get_harvard_data_loader(site=args.site,
                                          split=args.split,
                                          split_type=args.split_type,
                                          normalize_eeg_method=args.normalize_eeg_method,
                                          task=args.task,
                                          load_eeg=args.load_eeg,
                                          batch_size=1,
                                          num_workers=args.num_workers,
                                          max_eeg_sequence_length=UNIMODAL_TEXT_AND_EEG_FEATURES_BASELINE_CONFIG['max_eeg_sequence_length'])
    

    # test data loader
    for batch_idx, batch_dict in enumerate(test_data_loader):
        for key in batch_dict.keys():
            print(f'{key}: {batch_dict[key][0]}')
        break
    
    # Load Model
    model_name = args.model_name
    if args.use_accelerate:
        llm_pipeline = pipeline('text-generation', 
                        model=model_name, 
                        device_map="auto")
    else:
        llm_pipeline = pipeline('text-generation', 
                     model=model_name, 
                     device=args.device)
        
    
    # Generate Report
    for batch_idx, batch_dict in tqdm.tqdm(enumerate(test_data_loader)):
        try:
            deidentified_file_name = batch_dict['meta_data'][0]['DeidentifiedName(Reports)'].replace('.txt', '')
            # check if the report is already generated
            if os.path.exists(os.path.join(save_dir, 'generated_reports_json', f'GENERATED_REPORT_{deidentified_file_name}.json')):
                print(f'Report {deidentified_file_name} already generated')
                continue
            
            report_generation_task_prompt = batch_dict['generated_prompt'][0]
            
            # Generate Report
            generated_reports = llm_pipeline(report_generation_task_prompt, 
                                            max_new_tokens=args.max_new_tokens, 
                                            return_full_text=False)
            
            generated_report_temp = generated_reports[0]['generated_text']

            # Save Generated Report
            with open(os.path.join(save_dir, 'generated_reports_txt', f'GENERATED_REPORT_{deidentified_file_name}.txt'), 'w') as f:
                f.write(generated_report_temp)
                    
            generated_report_json = extract_json(generated_report_temp)

            # second attempt to parse
            if generated_report_json is None:
                print(f'RETRY: Error extracting JSON from generated report {deidentified_file_name}')
                clean_generated_report_text = clean_generation_for_json_parsing(generated_report_temp)
                generated_report_json = extract_json(clean_generated_report_text)

                
                    
            if generated_report_json is None:
                print(f'Error extracting JSON from generated report {deidentified_file_name}')
                # print(generated_report_temp)
                error_generated_reports['error_generated_reports'].append({'DeidentifiedName(Reports)': deidentified_file_name, 'generated_report': generated_report_temp})
            else:
                # Save Generated Report
                with open(os.path.join(save_dir, 'generated_reports_json', f'GENERATED_REPORT_{deidentified_file_name}.json'), 'w') as f:
                    json.dump(generated_report_json, f)
                

            
            if batch_idx%10 == 0:
                # Save Error Generated Reports
                with open(os.path.join(save_dir, 'error_generated_reports.json'), 'w') as f:
                    json.dump(error_generated_reports, f)
        
        except torch.cuda.OutOfMemoryError as e:
            # Handle CUDA OOM error
            print(f"\n{'='*70}")
            print(f"CUDA OUT OF MEMORY ERROR for report: {deidentified_file_name}")
            print(f"Error: {str(e)}")
            print(f"Skipping this sample and continuing...")
            print(f"{'='*70}\n")
            
            # Log the OOM sample
            error_generated_reports['error_generated_reports'].append({
                'DeidentifiedName(Reports)': deidentified_file_name,
                'batch_idx': batch_idx,
                'error': str(e),
                'prompt_length': len(report_generation_task_prompt)
            })
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            continue
        except Exception as e:
            print(f'Error generating report {deidentified_file_name}')
            print(f'Error: {str(e)}')
            error_generated_reports['error_generated_reports'].append({'DeidentifiedName(Reports)': deidentified_file_name, 'generated_report': generated_report_temp})
            continue
        # break
    
    print(f'Total number of error generated reports: {len(error_generated_reports["error_generated_reports"])}')
    print(f'Total number of generated reports in txt: {len(os.listdir(os.path.join(save_dir, "generated_reports_txt")))}')
    print(f'Total number of generated reports in json: {len(os.listdir(os.path.join(save_dir, "generated_reports_json")))}')
    print('Completed!')