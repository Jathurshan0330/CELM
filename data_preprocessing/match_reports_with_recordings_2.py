import pandas as pd
import glob
import os
import shutil
import tqdm
import subprocess
import argparse

def download_from_s3(s3_path, local_path):
    """Download files from S3 using aws s3 cp command"""
    try:
        cmd = ['aws', 's3', 'cp', s3_path, local_path, '--recursive']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'Error downloading from S3: {e}')
        print(f'stderr: {e.stderr}')
        return False
    
parser = argparse.ArgumentParser()
parser.add_argument('--site', type=str, default='S0001', help='Site ID')
args = parser.parse_args()

site = args.site
meta_data_path = f'[PATH_TO_EEG_REPORTS_HEEDB_METADATA]/{site}_EEG__reports_findings.csv'
recordings_data_path = f'[PATH_TO_EEG_RECORDINGS_AWS_S3_BUCKET]/{site}/'
processed_reports_path = f'[PATH_TO_PROCESSED_REPORTS]/{site}/Meta-Llama-3-8B-Instruct_llm_extracted_reports'
processed_reports_meta_data_path = f'[PATH_TO_PROCESSED_REPORTS]/{site}/neurology_report_metadata_EEG_{site}.csv'
matched_eeg_report_recording_save_path = f'[PATH_TO_MATCHED_EEG_RECORDINGS_REPORT]/{site}'
start_index = 0 #6000 #0
os.makedirs(matched_eeg_report_recording_save_path, exist_ok=True)


eeg_reports_meta_data_csv = pd.read_csv(meta_data_path)
# remove duplicates from eeg_reports_csv
eeg_reports_meta_data_csv = eeg_reports_meta_data_csv.drop_duplicates(subset=['DeidentifiedName(Reports)', 'BDSPPatientID','SessionID'])
print('EEG reports meta data shape:',eeg_reports_meta_data_csv.shape)

processed_reports_meta_data_csv = pd.read_csv(processed_reports_meta_data_path)
print('Processed reports meta data shape:',processed_reports_meta_data_csv.shape)
# processed_reports_meta_data_csv

# print number of files in processed_reports_path
print('Number of report files in processed_reports_path:',len(os.listdir(processed_reports_path)))

f"""
1. for each row in processed_reports_meta_data_csv, check if the file exists in the eeg_reports_meta_data_csv
2. if the file exists, get the corresponding all BDSPPatientID, SessionID
3. create a new directory in processed report basename and copy corresponding recordings from recordings_data_path and processed reports from processed_reports_path to the new directory
4. Create a overall description pandas dataframe with the following columns:
    - DeidentifiedName(Reports)
    - BDSPPatientID
    - SessionIDs: list of session ids 
    - NumberOfSessions
    - MatchedSavePath
    - VisitTypeDSC
    - ProcedureDSC
    - RecordType
    - AgeAtVisit
    - SexDSC
    - ProcedureDSC(Reports)
"""

description_df = pd.DataFrame(columns=['DeidentifiedName(Reports)', 'BDSPPatientID', 'SessionIDs','NumberOfSessions', 'MatchedSavePath', 'VisitTypeDSC', 'ProcedureDSC', 'RecordType', 'AgeAtVisit', 'SexDSC', 'ProcedureDSC(Reports)' ])
processed_reports_meta_data_csv = processed_reports_meta_data_csv.iloc[start_index:]
print(processed_reports_meta_data_csv.shape)

for index, row in tqdm.tqdm(processed_reports_meta_data_csv.iterrows()):
    file = row['DeidentifiedName']
    if file in eeg_reports_meta_data_csv['DeidentifiedName(Reports)'].values:
        # print(file)
        # check if the processed report exists in the processed report folder
        if not os.path.exists(os.path.join(processed_reports_path, file.replace('.txt', '.json'))):
            print(f'{file} does not exist in the processed report folder')
            continue
        
        matched_save_path = file.replace('.txt', '')
        matched_save_path = os.path.join(matched_eeg_report_recording_save_path, matched_save_path)
        
        temp_df = eeg_reports_meta_data_csv[eeg_reports_meta_data_csv['DeidentifiedName(Reports)'] == file]
        if os.path.exists(os.path.join(matched_save_path, 'eeg_recordings')):
            # check if the files exists in the matched_save_path
            if os.path.exists(os.path.join(matched_save_path, 'eeg_recordings', f'sub-{site}{temp_df["BDSPPatientID"].values[0]}_ses-{temp_df["SessionID"].values[0]}','eeg')):
                
                if len(temp_df) == len(os.listdir(os.path.join(matched_save_path, 'eeg_recordings'))):
                    print(f'skipping {file} because it already exists in the matched_save_path')
                    continue

        
        os.makedirs(matched_save_path, exist_ok=True)
        os.makedirs(os.path.join(matched_save_path, 'eeg_recordings'), exist_ok=True)
        
        # copy the corresponding report from recordings_data_path to the matched_save_path
        temp_report_path = os.path.join(processed_reports_path, file.replace('.txt', '.json'))
        shutil.copy(temp_report_path, os.path.join(matched_save_path, file.replace('.txt', '.json')))
        
        # copy the corresponding recordings from recordings_data_path to the matched_save_path
        for index_2, row_2 in temp_df.iterrows():
            session_id = row_2['SessionID']
            patient_id = row_2['BDSPPatientID']
            patient_id = f'sub-{site}{patient_id}'
            
            s3_session_path = os.path.join(recordings_data_path, patient_id, f'ses-{session_id}/')
            local_session_path = os.path.join(matched_save_path, 'eeg_recordings', f'{patient_id}_ses-{session_id}')
            os.makedirs(local_session_path, exist_ok=True)
            
            # download the session from s3 to local
            # cmd = ['aws', 's3', 'ls', s3_session_path]
            # result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # print(result.stdout)
            success = download_from_s3(s3_session_path, local_session_path)
            
            #check if the temp_eeg_recording_folder exists
            if not success:
                print(f'{s3_session_path} download failed')
                continue
            
        description_df = pd.concat([description_df, pd.DataFrame({
            'DeidentifiedName(Reports)': file,
            'BDSPPatientID': temp_df['BDSPPatientID'].values.tolist(),
            'SessionIDs': temp_df['SessionID'].values.tolist(),
            'NumberOfSessions': len(temp_df['SessionID'].values.tolist()),
            'MatchedSavePath': matched_save_path,
            'VisitTypeDSC': row['VisitTypeDSC'],
            'ProcedureDSC': row['ProcedureDSC'],
            'RecordType': row['RecordType'],
            'AgeAtVisit':temp_df['AgeAtVisit'].values.tolist(),
            'SexDSC':temp_df['SexDSC'].values.tolist(),
            'ProcedureDSC(Reports)':temp_df['ProcedureDSC(Reports)'].values.tolist(),
        })], ignore_index=True)
        
        # break

print(description_df.shape)
description_df.to_csv(os.path.join(matched_eeg_report_recording_save_path, f'{site}_matched_eeg_report_recording_description.csv'), index=False)
print(f'{site}_matched_eeg_report_recording_description.csv saved')