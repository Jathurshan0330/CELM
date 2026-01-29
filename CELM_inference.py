# import warnings
# warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import math
from typing import Dict, List, Optional
from datetime import datetime
import tqdm
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

from configs.default_configs import  EEG_LLM_PROJECTION_ONLY_CONFIG
from dataset.eeg_report_data_loader import get_harvard_data_loader
from eeg_llm.eeg_llm import create_eeg_llm, EEGLLM

from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from utils.utils import seed_everything, extract_json, clean_generation_for_json_parsing

# print("CUDA_VISIBLE_DEVICES env:", os.environ.get("CUDA_VISIBLE_DEVICES"))


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
            
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get list of trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]

          

def load_checkpoint(
    model: EEGLLM,
    checkpoint_path: str
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.projector.load_state_dict(checkpoint["projector_state_dict"])
    model.start_eeg_token.data = checkpoint["start_eeg_token"]
    model.end_eeg_token.data = checkpoint["end_eeg_token"]
    model.eeg_session_separator_token.data = checkpoint["eeg_session_separator_token"]
    

    return checkpoint, model

    
    
@torch.no_grad()
def evaluate(
    model: EEGLLM,
    test_loader,
    device: str,
    ) -> Dict[str, float]:
    
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    

   
    
    for batch_idx, batch in tqdm.tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
        
        
        eeg_data = batch["eeg_segments"]
        prompts = batch["generated_prompt"]
        labels = batch["labels"]
        
        outputs = model(eeg_data, prompts, labels)
        # total_loss += outputs.loss.item()
        # num_batches += 1
        loss_val = outputs.loss.detach()
        
       

        total_loss += loss_val.item()
        num_batches += 1
        

        # break
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity
    }
    



            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_encoder_model", type=str, default="cbramod")
    parser.add_argument("--llm_model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--projector", type=str, default="linear")
    parser.add_argument("--training_mode", type=str, default="projection_only")
    
    # Dataset Parameters
    parser.add_argument('--site', type=str, default='S0002')
    parser.add_argument('--split_type', type=str, default='random_split_data_by_patient')
    parser.add_argument('--normalize_eeg_method', type=str, default='div_by_100')
    parser.add_argument('--task', type=str, default='eeg_llm_projection_only')
    parser.add_argument('--num_workers', type=int, default=8)
    
    # experiment parameters
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='test')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--combine_k', type=int, default=None)
    parser.add_argument('--drop_last', action='store_true')
    
    # limit max sequence length of eeg data
    parser.add_argument('--max_eeg_sequence_length', type=int, default=None)
    
    
    args = parser.parse_args()
    
    # load config yaml
    model_save_name = args.llm_model.split('/')[-1]
    with open(f'./configs/training_configs/{args.training_mode}/{args.eeg_encoder_model}_{model_save_name}_{args.projector}.yaml', 'r') as f:
        model_training_config = yaml.safe_load(f)
    
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    
    
    # seed everything
    seed_everything(model_training_config['seed'])
    
    # Create Save Directory
    if args.experiment_name:
        save_dir = os.path.join(EEG_LLM_PROJECTION_ONLY_CONFIG['save_dir'], args.experiment_name, model_training_config['checkpoint_dir'])
    else:
        save_dir = os.path.join(EEG_LLM_PROJECTION_ONLY_CONFIG['save_dir'], model_training_config['checkpoint_dir'])
    checkpoint_result_save_folder = args.checkpoint_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(save_dir, f'inference_results_{args.site}', checkpoint_result_save_folder)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_reports_txt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generated_reports_json'), exist_ok=True)
    
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
    
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading test datasets...")
    test_data_loader = get_harvard_data_loader(site=args.site,
                                          split='test',
                                          split_type=args.split_type,
                                          normalize_eeg_method=args.normalize_eeg_method,
                                          task=args.task,
                                          load_eeg=True,
                                          batch_size=1,
                                          num_workers=args.num_workers,
                                          combine_k=args.combine_k,
                                          drop_last=args.drop_last)
    
    
    # test data loader
    for batch_idx, batch_dict in enumerate(test_data_loader):
        for key in batch_dict.keys():
            print(f'{key}: {batch_dict[key][0]}')
        break
    print(f"Testing samples: {len(test_data_loader.dataset)}")
    
    # -------------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading model...")

    eeg_llm = create_eeg_llm(
        config=model_training_config,
        device=device,
    )
    # Load Checkpoint
    print(f"Resuming from last checkpoint: {args.checkpoint_path}")
    checkpoint, eeg_llm = load_checkpoint(
        model=eeg_llm,
        checkpoint_path=args.checkpoint_path
    )
    
    eeg_llm.eval()
    eeg_llm.to(device)
    
    if hasattr(eeg_llm, "llm") and hasattr(eeg_llm.llm, "config"):
        print("Disabling cache for LLM")
        eeg_llm.llm.config.use_cache = False  # critical for training
    # after eeg_llm created (before accelerator.prepare)
    if hasattr(eeg_llm, "llm") and hasattr(eeg_llm.llm, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing for LLM")
        eeg_llm.llm.gradient_checkpointing_enable()
        
        
    # save model summary
    with open(os.path.join(save_dir, f'model_summary.txt'), 'w') as f:
        f.write(str(eeg_llm))
    
    # Print parameter counts
    param_counts = count_parameters(eeg_llm)
    print(f"\nParameter counts:")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable %: {100 * param_counts['trainable'] / param_counts['total']:.4f}%")
    
    

    print("\n[4/5] Starting evaluation.. Generating reports.")
        

    for batch_idx, batch_dict in tqdm.tqdm(enumerate(test_data_loader)):
        deidentified_file_name = batch_dict['meta_data'][0]['DeidentifiedName(Reports)'].replace('.txt', '')
        # check if the report is already generated
        if os.path.exists(os.path.join(save_dir, 'generated_reports_json', f'GENERATED_REPORT_{deidentified_file_name}.json')):
            print(f'Report {deidentified_file_name} already generated')
            continue
        
        report_generation_task_prompt = batch_dict['generated_prompt'][0]
        
        # Generate Report
        with torch.no_grad():
            generated_reports = eeg_llm.generate(eeg_data=batch_dict['eeg_segments'],
                                                 prompts=batch_dict['generated_prompt'],
                                                 max_new_tokens=2048)
        
        generated_report_temp = generated_reports[0]

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
        
        # break
    
    print(f'Total number of error generated reports: {len(error_generated_reports["error_generated_reports"])}')
    print(f'Total number of generated reports in txt: {len(os.listdir(os.path.join(save_dir, "generated_reports_txt")))}')
    print(f'Total number of generated reports in json: {len(os.listdir(os.path.join(save_dir, "generated_reports_json")))}')
    print('Completed!')
                
            
            