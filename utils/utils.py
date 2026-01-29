import os
import random
import numpy as np
import torch
import re
import json
from scipy.signal import welch
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

def seed_everything(seed=5):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def extract_json(text):
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object
    json_match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    return None


def clean_generation_for_json_parsing(generated_report_text):
    generated_report_text_temp = generated_report_text
    try:
        # generated_report_text = generated_report_text.split(']}')[0] + ']}\n'
        if '```' in generated_report_text:
            generated_report_text = generated_report_text.split('```')[1].split('```')[0].strip()
        else:
            # split at the first { and take all the text after it
            generated_report_text = generated_report_text.split('{',1)[1]
            end = generated_report_text.rfind('}')
            generated_report_text = generated_report_text[:end]
            generated_report_text = '{' + generated_report_text + '}'
            
        # check if generated_report_text ends.with a }
        if not generated_report_text.endswith('}'):
            if ']' not in generated_report_text:
                generated_report_text = generated_report_text + ']}'
            else:
                generated_report_text = generated_report_text + '}'
                
        if ']' not in generated_report_text:
            generated_report_text = generated_report_text[:-1] + ']}'
        # generated_report_text = [x for x in generated_report_text if 'json' in x][0]
        generated_report_text = '```'+generated_report_text+'```'
        
        generated_report_text = generated_report_text.replace('...', '')
        
        start_json = generated_report_text.split('[')[0]
        end_json = generated_report_text.split(']')[1]
        
        temp_generated_report_text = generated_report_text.replace(start_json, '').replace(end_json, '')
        generated_list = []
        
        temp_generated_report_text = temp_generated_report_text.split('{')

        
        for item in temp_generated_report_text:
            if 'section_name' in item:
                item = item.replace('}', '')
                item = item.replace('{', '')
                item = item.replace(']', '')
                item = '{' + item + '}'
                item = item.replace('\n', ' ')
                generated_list.append(item)

        generated_report_text = start_json + '[' + ','.join(generated_list) + ']' + end_json
        return generated_report_text
    except:
        return generated_report_text_temp
    
    
    


def bandpower_segments(
    X: np.ndarray,
    fs: float,
    num_seg_to_combine_for_pooling:  int = None,
    bands=None,
    nperseg=None,
    noverlap=None,
    window="hann",
    detrend="constant",
    scaling="density",
    relative: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute frequency band power for EEG segments.

    Args:
        X: EEG data, shape (segments, channels, time)
        fs: sampling rate (Hz)
        bands: list of (name, f_low, f_high). If None, uses common EEG bands.
        nperseg, noverlap, window, detrend, scaling: passed to scipy.signal.welch
        relative: if True, return relative band power (divide by total power in band range)
        eps: numerical stability

    Returns:
        band_powers: shape (segments, channels, n_bands)
        (order matches `bands`)
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (segments, channels, time). Got {X.shape}")

    
    if bands is None:
        bands = [
            ("delta", 0.5, 4.0),
            ("theta", 4.0, 8.0),
            ("alpha", 8.0, 13.0),
            ("beta", 13.0, 30.0),
            ("gamma", 30.0, 80.0),
        ]

    S, C, T = X.shape
    # Pool segments if requested
    if num_seg_to_combine_for_pooling is not None and num_seg_to_combine_for_pooling > 1:
        if num_seg_to_combine_for_pooling > S:
            num_seg_to_combine_for_pooling = S
        
        # Calculate number of pooled segments
        num_pooled_segs = S // num_seg_to_combine_for_pooling
        # Trim to make evenly divisible
        S_trimmed = num_pooled_segs * num_seg_to_combine_for_pooling
        X_trimmed = X[:S_trimmed]
        
        # Reshape and concatenate along time axis: (num_pooled_segs, channels, num_seg_to_combine_for_pooling * time)
        X = X_trimmed.reshape(num_pooled_segs, num_seg_to_combine_for_pooling, C, T)
        X = X.transpose(0, 2, 1, 3).reshape(num_pooled_segs, C, num_seg_to_combine_for_pooling * T)
        
        S = num_pooled_segs
        T = num_seg_to_combine_for_pooling * T
    
    # print(X.shape)
    
    if nperseg is None:
        # reasonable default: 2 seconds or as much as available (cap to T)
        nperseg = int(min(T, max(256, 2 * fs)))

    # Welch over last axis (time) for all segments/channels at once
    f, Pxx = welch(
        X,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
        axis=-1,
    )  # Pxx shape: (S, C, F)

    # Optional total power (within min/max band range) for relative power
    if relative:
        total_power = np.trapz(Pxx, f, axis=-1)  # (S, C)
    # (S, C)

    out = np.zeros((S, C, len(bands)), dtype=np.float32)

    for i, (_, f_low, f_high) in enumerate(bands):
        mask = (f >= f_low) & (f < f_high)
        if not np.any(mask):
            out[..., i] = 0.0
            continue
        bp = np.trapz(Pxx[..., mask], f[mask], axis=-1)  # (S, C)
        if relative:
            bp = bp / (total_power + eps)
        out[..., i] = bp.astype(np.float32)
    
    # convert to log scale
    out = 10 * np.log10(out + eps) #10 * np.log10(P + 1e-12)
    out = np.round(out,1)
    return out


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results