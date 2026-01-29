import pandas as pd
import glob
import os
import mne
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--site', type=str, default='S0001', help='Site ID')
args = parser.parse_args()


site = args.site
matched_eeg_report_recording_save_path = f'[PATH_TO_MATCHED_EEG_RECORDINGS_REPORT]/{site}'




# At the beginning of main
log_file = os.path.join(matched_eeg_report_recording_save_path, f'processing_log_{site}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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

log_f = open(log_file, 'w')
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

print(f"Log file: {log_file}")


note_sub_dirs = os.listdir(matched_eeg_report_recording_save_path)
print(note_sub_dirs[0])
# remove .csv
note_sub_dirs = [folder for folder in note_sub_dirs if not folder.endswith('.csv')]
note_sub_dirs = [folder for folder in note_sub_dirs if not folder.endswith('.txt')]
note_sub_dirs = [folder for folder in note_sub_dirs if folder != 'bad_samples']
print(len(note_sub_dirs))

# sort the note_sub_dirs by base name
note_sub_dirs.sort(key=lambda x: int(x.split('_')[1]))
## Just to process faster for S0002
note_sub_dirs = note_sub_dirs[5000:]
print(len(note_sub_dirs))
## Just to process faster for S0002
# # remove files thats are < 13536237358
# note_sub_dirs = [x for x in note_sub_dirs if int(x.split('_')[1]) >=13536237358]
# print(f'first dir {print(note_sub_dirs[0])}')


EEG_CHANNELS = [
            'C3', 'C4', 'O1', 'O2', 'Cz',
            'F3', 'F4', 'F7', 'F8', 'Fz',
            'Fp1', 'Fp2', 'Fpz',
            'P3', 'P4', 'Pz',
            'T3', 'T4', 'T5', 'T6',
            'A1', 'A2'
        ]
RESAMPLING_RATE = 200
BAND = (0.1, 75)
NOTCH_FREQ = 60
SEGMENT_LENGTH = 10
OVERLAP = 0.0


def is_directory_processed(note_sub_dir_path):
    """Check if a directory has already been processed"""
    processed_eeg_path = os.path.join(note_sub_dir_path, 'processed_eeg')
    if not os.path.exists(processed_eeg_path):
        return False
    eeg_recordings_path = glob.glob(os.path.join(note_sub_dir_path, 'eeg_recordings', '*'))
    for eeg_record in eeg_recordings_path:
        eeg_edf_files = glob.glob(os.path.join(eeg_record, 'eeg', '*.edf'))[0]
        raw = mne.io.read_raw_edf(eeg_edf_files, preload=True, verbose=False)
        fs = raw.info['sfreq']
        
        raw.resample(RESAMPLING_RATE, verbose=False)
        fs = RESAMPLING_RATE
        num_segments = (len(raw.times) - SEGMENT_LENGTH * fs) // (SEGMENT_LENGTH * fs * (1 - OVERLAP)) + 1
        eeg_record_base_name = os.path.basename(eeg_record)
        processed_eeg_save_path_temp = os.path.join(processed_eeg_path, eeg_record_base_name)
        if not os.path.exists(processed_eeg_save_path_temp):
            return False
        if len(os.listdir(processed_eeg_save_path_temp)) != num_segments:
            print(f'number of segments {num_segments} != number of pickles {len(os.listdir(processed_eeg_save_path_temp))} for {eeg_edf_files}')
            return False
    return True

def process_single_eeg_record(eeg_record, processed_eeg_save_path, 
                               EEG_CHANNELS, RESAMPLING_RATE, BAND, 
                               NOTCH_FREQ, SEGMENT_LENGTH, OVERLAP):
    """Process a single EEG recording"""
    try:
        eeg_record_base_name = os.path.basename(eeg_record)
        processed_eeg_save_path_temp = os.path.join(processed_eeg_save_path, eeg_record_base_name)
        os.makedirs(processed_eeg_save_path_temp, exist_ok=True)
        print(f'Processing {eeg_record_base_name}')
        # edf file
        eeg_edf_files = glob.glob(os.path.join(eeg_record, 'eeg', '*.edf'))
        if len(eeg_edf_files) == 0:
            print(f"No EDF file found in {eeg_record}")
            return 0
        
        eeg_edf_file = eeg_edf_files[0]
        
        # load the edf file
        raw = mne.io.read_raw_edf(eeg_edf_file, preload=True, verbose=False)
        
        # preprocess the raw data 
        raw.filter(l_freq=BAND[0], h_freq=BAND[1], verbose=False, n_jobs=1)
        raw.notch_filter(NOTCH_FREQ, verbose=False, n_jobs=1)
        
        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        channel_tsv_file = glob.glob(os.path.join(eeg_record, 'eeg', '*.tsv'))
        
        # remove bad channels
        if len(channel_tsv_file) > 0:
            channel_tsv = pd.read_csv(channel_tsv_file[0], sep='\t')
            good_channels = channel_tsv[channel_tsv['status'] == 'good']['name'].tolist()
            available_channels = [ch for ch in available_channels if ch in good_channels]
        
        # get signals for available channels
        raw.pick(available_channels)
        
        # get sampling rate
        fs = raw.info['sfreq']
        if fs != RESAMPLING_RATE:
            raw.resample(RESAMPLING_RATE, verbose=False, n_jobs=1)
            fs = RESAMPLING_RATE
            
        # get signals
        data_existing, times = raw[:, :]
        
        eeg_data = np.zeros((len(EEG_CHANNELS), len(times)))
        mean_eeg_data = np.zeros((len(EEG_CHANNELS)))
        std_eeg_data = np.zeros((len(EEG_CHANNELS)))
        
        for i, ch in enumerate(EEG_CHANNELS):
            if ch in available_channels:
                idx = available_channels.index(ch)
                eeg_data[i] = data_existing[idx]
                mean_eeg_data[i] = np.mean(eeg_data[i])
                std_eeg_data[i] = np.std(eeg_data[i])
        
        # segment the data into 10 seconds segments and save the data 
        seg_samples = int(SEGMENT_LENGTH * fs)
        step = int(seg_samples * (1 - OVERLAP))
        seg_count = 0
        eeg_record_base_name_temp = eeg_record_base_name.split('_')[0]
        
        # get the number of segments
        num_segs = (len(times) - seg_samples) // step + 1
        
        for start in range(0, len(times) - seg_samples + 1, step):
            end = start + seg_samples
            seg_data = eeg_data[:, start:end]
            
            # create a pickle file to save the data
            pickle_file_path = os.path.join(processed_eeg_save_path_temp, 
                                           f'seg_{seg_count}_{eeg_record_base_name_temp}.pkl')
            with open(pickle_file_path, 'wb') as f:
                pickle.dump({
                    'available_channels': available_channels, 
                    'mean_eeg_data': mean_eeg_data, 
                    'std_eeg_data': std_eeg_data, 
                    'signal': seg_data
                }, f)
            seg_count += 1
        
        assert seg_count == num_segs, f'Number of segments: {seg_count} != {num_segs}'
        print(f'Completed {eeg_record_base_name}')
        return seg_count
        
    except Exception as e:
        print(f"Error processing {eeg_record}: {str(e)}")
        return 0


def process_note_subdir(note_sub_dir, matched_eeg_report_recording_save_path,
                       EEG_CHANNELS, RESAMPLING_RATE, BAND, 
                       NOTCH_FREQ, SEGMENT_LENGTH, OVERLAP):
    """Process a single note subdirectory"""
    try:
        note_sub_dir_path = os.path.join(matched_eeg_report_recording_save_path, note_sub_dir)
        print(f'Processing {note_sub_dir} ===========================')
        # check if the directory is already processed
        if is_directory_processed(note_sub_dir_path):
            print(f"Directory {note_sub_dir_path} already processed")
            return note_sub_dir, 0
        
        # create a folder to save processed EEG signals
        processed_eeg_save_path = os.path.join(note_sub_dir_path, 'processed_eeg')
        os.makedirs(processed_eeg_save_path, exist_ok=True)
        
        eeg_recordings_path = glob.glob(os.path.join(note_sub_dir_path, 'eeg_recordings', '*'))
        
        total_segments = 0
        for eeg_record in eeg_recordings_path:
            seg_count = process_single_eeg_record(
                eeg_record, processed_eeg_save_path,
                EEG_CHANNELS, RESAMPLING_RATE, BAND,
                NOTCH_FREQ, SEGMENT_LENGTH, OVERLAP
            )
            total_segments += seg_count
        
        # number of folders in processed_eeg_save_path == number of eeg_recordings_path
        if len(os.listdir(processed_eeg_save_path)) != len(eeg_recordings_path):
            print(f"Number of folders in processed_eeg_save_path: {len(os.listdir(processed_eeg_save_path))} != {len(eeg_recordings_path)}")
            return note_sub_dir, 0
        print(f'Completed {note_sub_dir} ===========================')
        
        return note_sub_dir, total_segments
        
    except Exception as e:
        print(f"Error processing directory {note_sub_dir}: {str(e)}")
        return note_sub_dir, 0
    
    
if __name__ == '__main__':
    # Determine number of processes (use 80% of available CPUs)
    num_processes = 32 #32 #max(1, int(cpu_count() * 0.8))
    print(f"Using {num_processes} processes")
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_note_subdir,
        matched_eeg_report_recording_save_path=matched_eeg_report_recording_save_path,
        EEG_CHANNELS=EEG_CHANNELS,
        RESAMPLING_RATE=RESAMPLING_RATE,
        BAND=BAND,
        NOTCH_FREQ=NOTCH_FREQ,
        SEGMENT_LENGTH=SEGMENT_LENGTH,
        OVERLAP=OVERLAP
    )
    
    # Process in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_func, note_sub_dirs),
            total=len(note_sub_dirs),
            desc="Processing directories"
        ))
    
    # Print summary
    print("\n=== Processing Summary ===")
    for dir_name, seg_count in results:
        if seg_count > 0:
            print(f"{dir_name}: {seg_count} segments")
    
    total_segments = sum(seg_count for _, seg_count in results)
    
    # save error log
    error_log_path = os.path.join(matched_eeg_report_recording_save_path, 'error_log.txt')
    with open(error_log_path, 'w') as f:
        for dir_name, seg_count in results:
            if seg_count == 0:
                f.write(f"{dir_name}\n")
    
    print(f"\nTotal segments created: {total_segments}")