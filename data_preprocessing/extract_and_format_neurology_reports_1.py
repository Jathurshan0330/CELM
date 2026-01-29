import tqdm
import json
import pandas as pd
import os
import argparse
from report_extract_utils import get_llm_response, extract_json, check_llm_extraction
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--HEEDB_patients_path', type=str, default=None)
parser.add_argument('--site_id_filter', type=str, default=None)
parser.add_argument('--model_name', type=str, default="google/medgemma-4b-it")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--notes_path', type=str, default=None)
parser.add_argument('--num_repetitions', type=int, default=1)
parser.add_argument('--overwrite_existing_reports', type=bool, default=False)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=None)
args = parser.parse_args()

print(args)

pipe = pipeline("text-generation", model=args.model_name,device=args.device, max_new_tokens=4096)

data_path = args.data_path
save_path = args.save_path
notes_path = args.notes_path

if data_path is None:
    raise ValueError('data_path is required')
if save_path is None:
    raise ValueError('save_path is required')
if notes_path is None:
    raise ValueError('notes_path is required')

if '/' in args.model_name:
    save_model_name = args.model_name.split('/')[-1]
else:
    save_model_name = args.model_name

if args.site_id_filter is not None:
    save_path = os.path.join(save_path, args.site_id_filter)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, f'{save_model_name}_llm_extracted_reports'), exist_ok=True)
    
    
########################################################
# Read neurology report metadata and filter for EEG related visits in the interested site
########################################################

neurology_report_csv = pd.read_csv(os.path.join(data_path, 'I0001_Neurology_Reports_20thNovember.csv'))
print('number of records in neurology reports: ', neurology_report_csv.shape[0])
# get unique patient IDs
unique_patient_ids_neurology_reports = neurology_report_csv['BDSPPatientID'].unique()
print('number of unique patient IDs in neurology reports: ', len(unique_patient_ids_neurology_reports))

if args.site_id_filter is not None:
    print('Filtering for site_id: ', args.site_id_filter)
    heedb_patients = pd.read_csv(f'{args.HEEDB_patients_path}/HEEDB_patients.csv') 
    site_id_filter_patients = heedb_patients[heedb_patients['SiteID'] == args.site_id_filter]
    print('number of patients in site_id: ', site_id_filter_patients.shape[0])
    # filter patientes with matched EEG reports
    site_id_filter_patients_with_matched_eeg_reports = site_id_filter_patients[site_id_filter_patients['MatchedEEGReports'] == 'Y']
    print('number of patients with matched EEG reports: ', site_id_filter_patients_with_matched_eeg_reports.shape[0])
    # filter neurology reports for site_id_filter_patients_with_matched_eeg_reports
    neurology_report_csv= neurology_report_csv[neurology_report_csv['BDSPPatientID'].isin(site_id_filter_patients_with_matched_eeg_reports['BDSPPatientID'])]
    print('number of records in neurology reports: after filtering for EEG related VisitTypeDSC and site_id_filter: ', neurology_report_csv.shape[0])
    print('number of unique patient IDs in neurology reports: after filtering for EEG related VisitTypeDSC and site_id_filter: ', len(neurology_report_csv['BDSPPatientID'].unique()))

# Filter neurology reports for EEG and ROUTINE EEG  visits
neurology_report_csv_EEG = neurology_report_csv[neurology_report_csv['VisitTypeDSC'].isin(['EEG', 
                                                                                           'ROUTINE EEG',
                                                                                           'EEG SLEEP',
                                                                                           'PORTABLE EEG',
                                                                                           'COMPLEX EEG',
                                                                                           'SLEEP EEG',
                                                                                           '24H EEG',
                                                                                           'EEG LONG TERM MONITORING',
                                                                                           'AWAKE EEG',
                                                                                           'EEG - SLEEP DEPRIVED',
                                                                                           '48 HOUR AMBULATORY EEG WITH VIDEO',
                                                                                           'EEG VISUAL EVOKE POTENTIAL',
                                                                                           '24 HOUR VIDEO EEG',
                                                                                           'EEG AUD EVOKE (BRAIN STEM)',
                                                                                           '72 HOUR VIDEO EEG',
                                                                                           'SLEEP/AWAKE EEG',
                                                                                           '48 HOUR VIDEO EEG',
                                                                                           '24 HOUR AMBULATORY EEG WITH VIDEO',
                                                                                           '72 HOUR AMBULATORY EEG WITH VIDEO',
                                                                                           '4 HOUR VIDEO EEG',
                                                                                           'EXTENDED EEG',
                                                                                           'EEG SENSORY EVOKE POT UPP',
                                                                                           'EEG SENSORY EVOKE POT LWR',
                                                                                           'EMU',
                                                                                           ])]

print('number of records in neurology reports: after filtering for EEG related VisitTypeDSC: ', neurology_report_csv_EEG.shape[0])
# save visit type distribution
visit_type_distribution = neurology_report_csv_EEG['VisitTypeDSC'].value_counts()
print(visit_type_distribution)

print('number of unique patient IDs in neurology reports: after filtering for EEG related VisitTypeDSC: ', len(neurology_report_csv_EEG['BDSPPatientID'].unique()))

# save neurology_report_csv_EEG metadata
neurology_report_csv_EEG.to_csv(os.path.join(save_path, f'neurology_report_metadata_EEG_{args.site_id_filter}.csv'), index=False)

########################################################
# Prompt for EEG sections extraction
########################################################

task_prompt_EEG = """You are a precise text extraction system. Your ONLY job is to locate and copy text exactly.

**Task:** Extract the [SECTION_NAME] section from the neurology report below.

**Rules (STRICTLY ENFORCED):**
1. Find the line starting with "{section_name}:" (case-insensitive)
2. Extract ALL text from after the colon until the next section header or end of document
3. Section headers are uppercase words followed by a colon (e.g., "INDICATION:", "METHOD:", "DETAIL:")
4. Copy the text CHARACTER-FOR-CHARACTER - no paraphrasing, no corrections, no modifications
5. Preserve all original formatting, typos, asterisks, timestamps, and line breaks
6. If "{section_name}:" does not exist in the report, return exactly: None

**Output:** Return ONLY this JSON structure, with no preamble, explanation, or markdown:
```json
{"section_text": "complete unmodified text of that section"}
```

Now process the following neurology report for [SECTION_NAME] section:

**REPORT TO PROCESS:**

[REPORT_TEXT]
"""


########################################################
# Extract and format neurology reports
########################################################
failed_eeg_sections_notes=[]
failed_clinical_sections_notes=[]

sections_dict = {

    'EEG_sections': [ 'details:', 'detail:','epileptiform abnormalities:',"sleep:","background:","eeg classification:","summary of the findings:",
                        "clinical significance:","eeg artifacts:","marked events and other clinical events:" ,"impression:",
                        "interictal epileptiform abnormalities:", "events/seizures:", "seizures:", 
                        "description:", "electroencephalogram description:", "eeg description:", "interpretation:",
                        "background activity:", "other activity:", 
                        ],
    
    
    'Clincal_sections': ['history:', 'clinical history:', "patient's clinical history:", "history/reason for monitoring:",
                            'indication:', 'comparison:', "indication for eeg:",
                            "clinical indication:", "reason for study:"
                            ],

}
for i in tqdm.tqdm(range(neurology_report_csv_EEG.shape[0])):
    if i < args.start_index:
        continue
    if args.end_index is not None and i > args.end_index:
        break
    note_file_name = neurology_report_csv_EEG['DeidentifiedName'].iloc[i]
    year = neurology_report_csv_EEG['Year'].iloc[i]
    VisitTypeDSC = neurology_report_csv_EEG['VisitTypeDSC'].iloc[i]

    note_file_path = os.path.join(notes_path, str(year), note_file_name)
    save_note_file_name = note_file_name.replace('.txt', '.json')
    if os.path.exists(os.path.join(save_path, f'{save_model_name}_llm_extracted_reports', f'{save_note_file_name}')):# and not args.overwrite_existing_reports:
        # print(f'{save_note_file_name} already exists')
        continue

    with open(note_file_path, 'r') as f:
        note_text = f.read()
    
    
    note_text = note_text.lower()
    note_text = note_text.replace('\n', ' ')
    
    processed_report_dict = {'VisitTypeDSC': VisitTypeDSC,
                             'DeidentifiedName': note_file_name,
                             'Year': str(year),
                             'note_text': note_text}
    # try:
    eeg_sections = []
    existing_eeg_sections = []
    for section_name in sections_dict['EEG_sections']:
        if section_name in note_text:
            existing_eeg_sections.append(section_name)
        else:
            continue
        task_prompt_EEG_edited = task_prompt_EEG.replace('[SECTION_NAME]', section_name)
        task_prompt_EEG_edited = task_prompt_EEG_edited.replace('[REPORT_TEXT]', note_text)
        extract_eeg_sections_response = get_llm_response(pipe, task_prompt_EEG_edited)
        try:
            extract_eeg_sections_response_json = extract_json(extract_eeg_sections_response)
            temp_dict = {'section_name': section_name, 'section_text': extract_eeg_sections_response_json['section_text']}
            eeg_sections.append(temp_dict)
        except Exception as e:
            continue
    
    if len(eeg_sections) == 0:
        failed_eeg_sections_notes.append(note_file_name)

    processed_report_dict['EEG_section_llm_extractions'] = {"EEG_sections": eeg_sections}
    
    clinical_sections = []
    existing_clinical_sections = []
    for section_name in sections_dict['Clincal_sections']:
        if section_name in note_text:
            existing_clinical_sections.append(section_name)
        else:
            continue
        task_prompt_EEG_edited = task_prompt_EEG.replace('[SECTION_NAME]', section_name)
        task_prompt_EEG_edited = task_prompt_EEG_edited.replace('[REPORT_TEXT]', note_text)
        extract_eeg_sections_response = get_llm_response(pipe, task_prompt_EEG_edited)
        try:
            extract_eeg_sections_response_json = extract_json(extract_eeg_sections_response)
            temp_dict = {'section_name': section_name, 'section_text': extract_eeg_sections_response_json['section_text']}
            clinical_sections.append(temp_dict)
        except Exception as e:
            continue
    
    if len(clinical_sections) == 0:
        failed_clinical_sections_notes.append(note_file_name)

    processed_report_dict['extracted_eeg_section_names'] = existing_eeg_sections
    processed_report_dict['extracted_clinical_section_names'] = existing_clinical_sections
    processed_report_dict['patient_history_section_llm_extractions'] = {"CLINICAL_sections": clinical_sections}
    
    
    # Get extraction scores
    scores = {}
    if len(eeg_sections) > 0:
        scores['eeg_sections'] = check_llm_extraction(note_text, processed_report_dict['EEG_section_llm_extractions']['EEG_sections'], use_similarity=True)
    if len(clinical_sections) > 0:
        scores['clinical_sections'] = check_llm_extraction(note_text, processed_report_dict['patient_history_section_llm_extractions']['CLINICAL_sections'], use_similarity=True)
    processed_report_dict['extraction_scores'] = scores

    # save processed report
    with open(os.path.join(save_path, f'{save_model_name}_llm_extracted_reports', f'{save_note_file_name}'), 'w') as f:
        json.dump(processed_report_dict, f)
    
    
    
    # intermediate saves of failed
    if i%10==0:
        with open(os.path.join(save_path, f'{save_model_name}_failed_eeg_notes_{args.start_index}_{args.end_index}.txt'), 'w') as f:
            for note in failed_eeg_sections_notes:
                f.write(note + '\n')
        with open(os.path.join(save_path, f'{save_model_name}_failed_clinical_notes_{args.start_index}_{args.end_index}.txt'), 'w') as f:
            for note in failed_clinical_sections_notes:
                f.write(note + '\n')
        


# save failed notes
with open(os.path.join(save_path, f'{save_model_name}_failed_eeg_notes_{args.start_index}_{args.end_index}.txt'), 'w') as f:
    for note in failed_eeg_sections_notes:
        f.write(note + '\n')
with open(os.path.join(save_path, f'{save_model_name}_failed_clinical_notes_{args.start_index}_{args.end_index}.txt'), 'w') as f:
    for note in failed_clinical_sections_notes:
        f.write(note + '\n')

    
    
    
            
            