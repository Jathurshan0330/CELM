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

from utils.utils import seed_everything

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

          
def save_checkpoint(
    model: EEGLLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "projector_state_dict": model.projector.state_dict(),
        "start_eeg_token": model.start_eeg_token.data,
        "end_eeg_token": model.end_eeg_token.data,
        "eeg_session_separator_token": model.eeg_session_separator_token.data,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    
    torch.save(checkpoint, f'{save_path}/checkpoint_epoch_{epoch}.pt')
    
    if is_best:
        best_path = f'{save_path}/checkpoint_best_epoch_{epoch}.pt'
        torch.save(checkpoint, best_path)

def load_checkpoint(
    model: EEGLLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.projector.load_state_dict(checkpoint["projector_state_dict"])
    model.start_eeg_token.data = checkpoint["start_eeg_token"]
    model.end_eeg_token.data = checkpoint["end_eeg_token"]
    model.eeg_session_separator_token.data = checkpoint["eeg_session_separator_token"]
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint, model, optimizer, scheduler

def train_step(
    model: EEGLLM,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Optional[Accelerator] = None, 
    scaler: Optional[GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    step: int = 0,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    
    model.train()
    eeg_data = batch["eeg_segments"]
    prompts = batch["generated_prompt"]
    labels = batch["labels"]

    if accelerator is not None:
        with accelerator.accumulate(model):
            outputs = model(eeg_data, prompts, labels)
            loss = outputs.loss
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        return {
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"]
        }
    # Forward pass with optional mixed precision
    elif use_amp and scaler is not None:
        with autocast():
            outputs = model(eeg_data, prompts, labels)
            loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
                
    else:
        outputs = model(eeg_data, prompts, labels)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
    
    return {
        "loss": loss.item() * gradient_accumulation_steps,
        "lr": optimizer.param_groups[0]["lr"]
    }
    
    
@torch.no_grad()
def validate(
    model: EEGLLM,
    val_loader,
    device: str,
    max_batches: Optional[int] = None,
    desc: str = "Validation",
    is_main_process: bool = True,
    accelerator: Optional[Accelerator] = None,
    ) -> Dict[str, float]:
    
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    if accelerator:
        accelerator.wait_for_everyone()
    
    # tqdm only on main process
    if is_main_process:
        iterator = tqdm.tqdm(
            enumerate(val_loader),
            total=min(len(val_loader), max_batches) if max_batches else len(val_loader),
            desc=desc,
            leave=False,
            position=1,
        )
    else:
        iterator = enumerate(val_loader)
    
    for batch_idx, batch in iterator:
        if max_batches and batch_idx >= max_batches:
            break
        
        eeg_data = batch["eeg_segments"]
        prompts = batch["generated_prompt"]
        labels = batch["labels"]
        
        outputs = model(eeg_data, prompts, labels)
        # total_loss += outputs.loss.item()
        # num_batches += 1
        loss_val = outputs.loss.detach()
        
        if accelerator is not None:
            loss_val = accelerator.gather_for_metrics(loss_val).mean()

        total_loss += loss_val.item()
        num_batches += 1
        
        if is_main_process:
            iterator.set_postfix({"val_loss": f"{(total_loss / num_batches):.4f}"})
        
        # break
        
    if accelerator:
        accelerator.wait_for_everyone()

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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_accelerate', action='store_true')
    parser.add_argument('--resume_from_last_checkpoint', action='store_true')
    parser.add_argument('--combine_k', type=int, default=None)
    parser.add_argument('--drop_last', action='store_true')
    
    # limit max sequence length of eeg data
    parser.add_argument('--max_eeg_sequence_length', type=int, default=None)
    # limit max batch size of eeg data for validation
    parser.add_argument('--max_val_eeg_batch_size', type=int, default=None)
    
    args = parser.parse_args()
    
    # load config yaml
    model_save_name = args.llm_model.split('/')[-1]
    with open(f'./configs/training_configs/{args.training_mode}/{args.eeg_encoder_model}_{model_save_name}_{args.projector}.yaml', 'r') as f:
        model_training_config = yaml.safe_load(f)
    
    # Setup Accelerate if using multiple GPUs
    if args.use_accelerate:
        accelerator = Accelerator(
            gradient_accumulation_steps=model_training_config['gradient_accumulation_steps'],
            # mixed_precision="fp16" if model_training_config['use_amp'] else "no"
            mixed_precision="bf16" if model_training_config['use_amp'] else "no"
        )
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        print(f"Using Accelerate with {accelerator.num_processes} GPUs")
        print(f"[Rank {accelerator.process_index}] Using device: {device}")
        print(f"[Rank {accelerator.process_index}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"[Rank {accelerator.process_index}] torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"[Rank {accelerator.process_index}] torch.cuda.current_device(): {torch.cuda.current_device()}")

    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        is_main_process = True
    
    
    # seed everything
    seed_everything(model_training_config['seed'])
    
    # Create Save Directory
    if args.experiment_name:
        save_dir = os.path.join(EEG_LLM_PROJECTION_ONLY_CONFIG['save_dir'], args.experiment_name, model_training_config['checkpoint_dir'])
    else:
        save_dir = os.path.join(EEG_LLM_PROJECTION_ONLY_CONFIG['save_dir'], model_training_config['checkpoint_dir'])
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, model_training_config['log_dir'])
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
        # save args
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            f.write(str(args.__dict__))
            
        # save configs
        with open(os.path.join(save_dir, 'model_training_config.yaml'), 'w') as f:
            yaml.dump(model_training_config, f)
            
        # Setup logging
        log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        log_f = open(log_file, 'w')
        sys.stdout = Tee(sys.stdout, log_f)
        sys.stderr = Tee(sys.stderr, log_f)

        print("=" * 70)
        print("EEG-LLM Projection Only Training Script")
        print("=" * 70)
        print(f"Save directory: {save_dir}")
        print(f"Device: {device}")
        print(f"Arguments: {json.dumps(vars(args), indent=2)}")
        print(f"Model Training Config: {json.dumps(model_training_config, indent=2)}")
        print("=" * 70)
        
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    if is_main_process:
        print("\n[1/5] Loading datasets...")
        
    
    train_data_loader = get_harvard_data_loader(site=args.site,
                                          split='train',
                                          split_type=args.split_type,
                                          normalize_eeg_method=args.normalize_eeg_method,
                                          task=args.task,
                                          load_eeg=True,
                                          batch_size=model_training_config['batch_size'],
                                          num_workers=args.num_workers,
                                          combine_k=args.combine_k,
                                          drop_last=args.drop_last)

    val_data_loader = get_harvard_data_loader(site=args.site,
                                          split='val',
                                          split_type=args.split_type,
                                          normalize_eeg_method=args.normalize_eeg_method,
                                          task=args.task,
                                          load_eeg=True,
                                          batch_size=model_training_config['batch_size'],
                                          num_workers=args.num_workers,
                                          combine_k=args.combine_k,
                                          drop_last=args.drop_last)
    
    
    if is_main_process:
        for batch_idx, batch_dict in enumerate(train_data_loader):
            for key in batch_dict.keys():
                print(f'{key}: {batch_dict[key][0]}')
            break
        print(f"Training samples: {len(train_data_loader.dataset)}")
        print(f"Validation samples: {len(val_data_loader.dataset)}")
    
    # -------------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------------
    if is_main_process:
        print("\n[2/5] Loading model...")
        
   
    print("Loading model from scratch")
    eeg_llm = create_eeg_llm(
        config=model_training_config,
        device=device,
    )
    if hasattr(eeg_llm, "llm") and hasattr(eeg_llm.llm, "config"):
        print("Disabling cache for LLM")
        eeg_llm.llm.config.use_cache = False  # critical for training
    # after eeg_llm created (before accelerator.prepare)
    if hasattr(eeg_llm, "llm") and hasattr(eeg_llm.llm, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing for LLM")
        eeg_llm.llm.gradient_checkpointing_enable()
        
        
    # save model summary
    if is_main_process:
        with open(os.path.join(save_dir, f'model_summary.txt'), 'w') as f:
            f.write(str(eeg_llm))
    
    # Print parameter counts
    if is_main_process:
        param_counts = count_parameters(eeg_llm)
        print(f"\nParameter counts:")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Frozen: {param_counts['frozen']:,}")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Trainable %: {100 * param_counts['trainable'] / param_counts['total']:.4f}%")
    
    
    # -------------------------------------------------------------------------
    # Setup Optimizer and Scheduler
    # -------------------------------------------------------------------------
    if is_main_process:
        print("\n[3/5] Setting up optimizer and scheduler...")
    
    trainable_params = get_trainable_parameters(eeg_llm)
    optimizer = AdamW(
        trainable_params,
        lr=float(model_training_config['learning_rate']),
        weight_decay=model_training_config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate total training steps
    num_training_steps = len(train_data_loader) * model_training_config['epochs'] // model_training_config['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * model_training_config['warmup_ratio'])
    
    # Learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if is_main_process:
        print(f"Optimizer: AdamW (lr={model_training_config['learning_rate']}, wd={model_training_config['weight_decay']})")
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {num_warmup_steps}")
    
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler() if model_training_config['use_amp'] and not args.use_accelerate else None
    
    
    
    # -------------------------------------------------------------------------
    # Load checkpoint if resuming
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    if args.resume_from_last_checkpoint:
        # get last checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoint_files = [f for f in checkpoint_files if 'best' not in f]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_checkpoint_file = checkpoint_files[-1]
        last_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_file)
        print(f"Resuming from last checkpoint: {last_checkpoint_path}")
        
        
        checkpoint, eeg_llm, optimizer, scheduler = load_checkpoint(
                model=eeg_llm,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=last_checkpoint_path
            )
            
        start_epoch = checkpoint['epoch']+1
        global_step = checkpoint['step']
        best_val_loss = checkpoint['loss']
        if is_main_process:
            print(f"Resuming from epoch {start_epoch} with global step {global_step}")
            print(f"Best validation loss: {best_val_loss}")
    
    # -------------------------------------------------------------------------
    # Prepare if using accelerate
    # -------------------------------------------------------------------------
    if args.use_accelerate:
        eeg_llm, optimizer, train_data_loader, val_data_loader, scheduler = accelerator.prepare(
            eeg_llm, optimizer, train_data_loader, val_data_loader, scheduler
        )
    
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    if is_main_process:
        print("\n[4/5] Starting training...")
        
        
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "learning_rate": []
    }
    
    if args.resume_from_last_checkpoint:
        training_history = json.load(open(os.path.join(log_dir, f"training_history_epoch_{start_epoch-1}.json"), "r"))
        if is_main_process:
            print(f'training history loaded from {os.path.join(log_dir, f"training_history_epoch_{start_epoch-1}.json")}')
            print(training_history)
    
    for epoch in range(start_epoch, model_training_config['epochs']):
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{model_training_config['epochs']}")
            print("-" * 50)
            
        epoch_loss = 0.0
        num_batches = 0
        
        
        # Progress bar
        if is_main_process:
            pbar = tqdm.tqdm(
                enumerate(train_data_loader),
                total=len(train_data_loader),
                desc=f"Epoch {epoch + 1}/{model_training_config['epochs']}"
            )
        else:
            pbar = enumerate(train_data_loader)
            
        for batch_idx, batch in pbar:
            # try:
            #     should_skip = any(eeg_session.shape[0] > 1000
            #         for batch_eeg in batch['eeg_segments']
            #         for eeg_session in batch_eeg)
            # except Exception as e:
            #     should_skip = True
            #     if is_main_process:
            #         print(f"Error checking batch, skipping: {e}")
            
            # if args.use_accelerate:
            #     # Synchronize skip decision across all ranks
            #     skip_tensor = torch.tensor([1 if should_skip else 0], 
            #                                 dtype=torch.int32, 
            #                                 device=accelerator.device)
            #     skip_tensor = accelerator.gather(skip_tensor)
            #     should_skip = skip_tensor.max().item() > 0

            # if should_skip:
            #     if is_main_process:
            #         print(f"Skipping batch {batch_idx} (synced across ranks)")
            #     continue
            
            # any eeg data >1000 skip
            should_skip = any(eeg_session.shape[0] > 1000
                  for batch_eeg in batch['eeg_segments']
                  for eeg_session in batch_eeg)
            if args.use_accelerate:
                # broadcast skip decision so all ranks do the same thing
                t = torch.tensor(int(should_skip), device=accelerator.device)
                t = accelerator.gather(t).max()   # if any rank wants to skip -> all skip
                should_skip = bool(t.item())
                accelerator.wait_for_everyone()

            if should_skip:
                if is_main_process:
                    print("Skipping batch (synced across ranks)")
                continue
        
            # if any(eeg_session.shape[0] > 1000 for batch_eeg in batch['eeg_segments'] for eeg_session in batch_eeg):
            #     for batch_eeg in batch['eeg_segments']:
            #         for eeg_session in batch_eeg:
            #             if eeg_session.shape[0] > 1000:
            #                 print(f'skipping batch as it has eeg data >1000')
            #                 print(f'eeg_session.shape: {eeg_session.shape}')
            #                 continue
            
            #     print(f'skipping batch as it has eeg data >1000')
            #     continue
            
            # Training step
            metrics = train_step(
                model=eeg_llm,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                gradient_accumulation_steps=model_training_config['gradient_accumulation_steps'],
                step=global_step,
                max_grad_norm=model_training_config['max_grad_norm'],
                use_amp=model_training_config['use_amp'] and not args.use_accelerate,
                accelerator=accelerator if args.use_accelerate else None, 
            )
            
            if args.use_accelerate:
                accelerator.wait_for_everyone()
    
            
            epoch_loss += metrics["loss"]
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            if is_main_process:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}"
                })
            
            # Log periodically
            if is_main_process and global_step % model_training_config['log_every_n_steps'] == 0:
                avg_loss = epoch_loss / num_batches
                print(f"\n  Step {global_step}: loss={avg_loss:.4f}, lr={metrics['lr']:.2e}")
            
            # break
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
       
        # Epoch Validation
        # val_metrics = validate(eeg_llm, val_data_loader, device)
        val_metrics = validate(
                        eeg_llm,
                        val_data_loader,
                        device,
                        desc=f"Val {epoch+1}/{model_training_config['epochs']}",
                        is_main_process=is_main_process,
                        accelerator=accelerator if args.use_accelerate else None,
                        max_batches=args.max_val_eeg_batch_size,
                    )
        
        if is_main_process:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch+1}/{model_training_config['epochs']} Summary:")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Perplexity: {val_metrics['val_perplexity']:.2f}")
            print(f"{'=' * 70}")
            
            # Save training history
            training_history["train_loss"].append(avg_epoch_loss)
            training_history["val_loss"].append(val_metrics["val_loss"])
            training_history["val_perplexity"].append(val_metrics["val_perplexity"])
            training_history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            
            with open(os.path.join(log_dir, f"training_history_epoch_{epoch}.json"), "w") as f:
                json.dump(training_history, f, indent=2)
            
            # Save epoch checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                is_best = True
                best_val_loss = val_metrics["val_loss"]
            else:
                is_best = False
                
            save_checkpoint(
                model=eeg_llm.module if hasattr(eeg_llm, 'module') else eeg_llm,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=val_metrics["val_loss"],
                save_path=checkpoint_dir,
                is_best=is_best
            )
            

    if is_main_process:
        print("\n[5/5] Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        

                
            
            