# generate description_df# rerunning the script to get the description_df
import pandas as pd
import json
import os
import argparse
from multiprocessing import Pool, Manager
from functools import partial
from pathlib import Path

def process_single_report(index, row, eeg_reports_meta_data_csv, processed_reports_path, 
                         matched_eeg_report_recording_save_path, site):
    """
    Process a single report and return the row data for description_df.
    This function is designed to be parallelized.
    """
    file = row['DeidentifiedName']
    print(f'Processing {index} : {file}')
    
    # Check if file exists in eeg_reports_meta_data
    if file not in eeg_reports_meta_data_csv['DeidentifiedName(Reports)'].values:
        return None, f'Does not exist in {site}_EEG__reports_findings.csv', file
    
    # Check if the processed report exists
    matched_save_path = file.replace('.txt', '')
    matched_save_path = os.path.join(matched_eeg_report_recording_save_path, matched_save_path)
    
    temp_df = eeg_reports_meta_data_csv[eeg_reports_meta_data_csv['DeidentifiedName(Reports)'] == file]
    
    eeg_recordings_path = os.path.join(matched_save_path, 'eeg_recordings')
    if not os.path.exists(eeg_recordings_path):
        return None, 'Not_matched', file
    
    # Check if all sessions are matched
    num_sessions_in_data = len(temp_df)
    num_sessions_in_folder = len(os.listdir(eeg_recordings_path))
    
    if num_sessions_in_data != num_sessions_in_folder:
        status = 'Matched_but_not_all_sessions_matched'
    else:
        status = 'Matched_and_all_sessions_matched'
    
    # Collect session-specific information
    session_ids = []
    age_at_visit = []
    sex_dsc = []
    procedure_dsc_reports = []
    processed_EEG_paths = []
    
    for _, row_2 in temp_df.iterrows():
        session_id = row_2['SessionID']
        patient_id = row_2['BDSPPatientID']
        patient_id = f'sub-{site}{patient_id}'
        
        eeg_session_path = os.path.join(eeg_recordings_path, f'{patient_id}_ses-{session_id}')
        if os.path.exists(eeg_session_path):
            processed_eeg_path = os.path.join(matched_save_path, f'processed_eeg/{patient_id}_ses-{session_id}')
            
            # Check for .pkl files
            has_pkl = False

            for root, dirs, files in os.walk(processed_eeg_path):
                if any(f.endswith('.pkl') for f in files):
                    has_pkl = True
                    break

            
            if has_pkl:
                session_ids.append(str(session_id))
                age_at_visit.append(str(row_2['AgeAtVisit']))
                sex_dsc.append(str(row_2['SexDSC']))
                procedure_dsc_reports.append(str(row_2['ProcedureDSC(Reports)']))
                processed_EEG_paths.append(f'processed_eeg/{patient_id}_ses-{session_id}')
    
    # Load the processed report JSON
    temp_report_path = os.path.join(processed_reports_path, file.replace('.txt', '.json'))
    try:
        with open(temp_report_path, 'r') as f:
            temp_report = json.load(f)
    except:
        return None, 'Failed_to_load_json', file
    
    
    # Extract empty sections
    empty_eeg_sections = []
    empty_clinical_sections = []
    
    for section_name in temp_report.get('EEG_section_llm_extractions', {}).get('EEG_sections', []):
        if section_name.get('section_text', '') == '':
            empty_eeg_sections.append(section_name.get('section_name', ''))
    
    if 'Clinical_section_llm_extractions' in temp_report:
        for section_name in temp_report['Clinical_section_llm_extractions'].get('Clinical_sections', []):
            if section_name.get('section_text', '') == '':
                empty_clinical_sections.append(section_name.get('section_name', ''))
    
    # Create the row data
    row_data = {
        'DeidentifiedName(Reports)': file,
        'BDSPPatientID': temp_df['BDSPPatientID'].values[0],
        'SessionIDs': ','.join(session_ids),
        'NumberOfSessions': len(session_ids),
        'MatchedSavePath': matched_save_path,
        'VisitTypeDSC': row['VisitTypeDSC'],
        'ProcedureDSC': row['ProcedureDSC'],
        'RecordType': row['RecordType'],
        'AgeAtVisit': ','.join(age_at_visit),
        'SexDSC': ','.join(sex_dsc),
        'ProcedureDSC(Reports)': ','.join(procedure_dsc_reports),
        'Extracted_EEG_sections': ','.join(temp_report.get('extracted_eeg_section_names', [])),
        'Empty_EEG_sections': ','.join(empty_eeg_sections),
        'Number_of_Extracted_EEG_sections': len(temp_report.get('extracted_eeg_section_names', [])),
        'Number_of_Empty_EEG_sections': len(empty_eeg_sections),
        'Extracted_Clinical_sections': ','.join(temp_report.get('extracted_clinical_section_names', [])),
        'Empty_Clinical_sections': ','.join(empty_clinical_sections),
        'Number_of_Extracted_Clinical_sections': len(temp_report.get('extracted_clinical_section_names', [])),
        'Number_of_Empty_Clinical_sections': len(empty_clinical_sections),
        'Processed_EEG_Paths': ','.join(processed_EEG_paths)
    }
    
    return row_data, status, file



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default='S0002', help='site name')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel workers')
    args = parser.parse_args()
    
    site = args.site
    num_workers = args.num_workers
    
    # Define paths
    meta_data_path = f'[PATH_TO_EEG_REPORTS_HEEDB_METADATA]/{site}_EEG__reports_findings.csv'
    processed_reports_path = f'[PATH_TO_PROCESSED_REPORTS]/{site}/Meta-Llama-3-8B-Instruct_llm_extracted_reports'
    processed_reports_meta_data_path = f'[PATH_TO_PROCESSED_REPORTS]/{site}/neurology_report_metadata_EEG_{site}.csv'
    matched_eeg_report_recording_save_path = f'[PATH_TO_MATCHED_EEG_RECORDINGS_REPORT]/{site}'
    
    # Load data
    print("Loading metadata...")
    eeg_reports_meta_data_csv = pd.read_csv(meta_data_path)
    eeg_reports_meta_data_csv = eeg_reports_meta_data_csv.drop_duplicates(
        subset=['DeidentifiedName(Reports)', 'BDSPPatientID', 'SessionID']
    )
    print(f'EEG reports meta data shape: {eeg_reports_meta_data_csv.shape}')
    
    processed_reports_meta_data_csv = pd.read_csv(processed_reports_meta_data_path)
    
    # Define the output dataframe columns
    description_df_columns = [
        'DeidentifiedName(Reports)', 'BDSPPatientID', 'SessionIDs', 'NumberOfSessions', 
        'MatchedSavePath', 'VisitTypeDSC', 'ProcedureDSC', 'RecordType', 'AgeAtVisit', 
        'SexDSC', 'ProcedureDSC(Reports)', 'Extracted_EEG_sections', 'Empty_EEG_sections', 
        'Number_of_Extracted_EEG_sections', 'Number_of_Empty_EEG_sections', 
        'Extracted_Clinical_sections', 'Empty_Clinical_sections', 
        'Number_of_Extracted_Clinical_sections', 'Number_of_Empty_Clinical_sections', 
        'Processed_EEG_Paths'
    ]
    
    # Initialize failed reports tracking
    failed_reports = {
        'Not_matched': [],
        'Matched_but_not_all_sessions_matched': [],
        'Matched_and_all_sessions_matched': [],
        f'Does not exist in {site}_EEG__reports_findings.csv': [],
        'Failed_to_load_json': []
    }
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_report,
        eeg_reports_meta_data_csv=eeg_reports_meta_data_csv,
        processed_reports_path=processed_reports_path,
        matched_eeg_report_recording_save_path=matched_eeg_report_recording_save_path,
        site=site
    )
    
    # Process in parallel
    print(f"Processing {len(processed_reports_meta_data_csv)} reports with {num_workers} workers...")
    
    description_rows = []
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_func,
            [(i, row) for i, (_, row) in enumerate(processed_reports_meta_data_csv.iterrows())]
        )
    
    # Aggregate results
    print("Aggregating results...")
    for row_data, status, file in results:
        if row_data is not None:
            description_rows.append(row_data)
        if status in failed_reports:
            failed_reports[status].append(file)
    
    # Create dataframe from collected rows
    description_df = pd.DataFrame(description_rows, columns=description_df_columns)
    
    # Print statistics
    print(f"\nProcessing Statistics:")
    print(f"Not matched: {len(failed_reports['Not_matched'])}")
    print(f"Matched but not all sessions matched: {len(failed_reports['Matched_but_not_all_sessions_matched'])}")
    print(f"Matched and all sessions matched: {len(failed_reports['Matched_and_all_sessions_matched'])}")
    print(f"Does not exist in metadata: {len(failed_reports[f'Does not exist in {site}_EEG__reports_findings.csv'])}")
    print(f"Failed to load JSON: {len(failed_reports['Failed_to_load_json'])}")
    
    # Save description_df
    output_path = os.path.join(
        matched_eeg_report_recording_save_path, 
        f'{site}_matched_eeg_report_recording_description.csv'
    )
    description_df.to_csv(output_path, index=False)
    print(f"\nDescription DataFrame shape: {description_df.shape}")
    print(f"Saved to: {output_path}")
    
    # Optional: Save failed reports summary
    failed_reports_output = os.path.join(
        matched_eeg_report_recording_save_path,
        f'{site}_failed_reports_summary.json'
    )
    with open(failed_reports_output, 'w') as f:
        json.dump(failed_reports, f, indent=2)
    print(f"Failed reports summary saved to: {failed_reports_output}")


if __name__ == '__main__':
    main()