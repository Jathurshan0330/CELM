import os
import glob
import pandas as pd
import json
import argparse
import datetime
import sys

from configs import default_configs 
from evaluate_gen.evaluate_gen import Evaluator


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
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--results_saved_path', type=str)
    parser.add_argument('--site', type=str, default='S0002')
    parser.add_argument('--split_type', type=str, default='random_split_data_by_patient')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--ignore_perplexity', action='store_true', default=False)
    args = parser.parse_args()

    model_name = args.model_name
    site = args.site
    split_type = args.split_type
    split = args.split
    ignore_perplexity = args.ignore_perplexity
    
    # load the split_csv
    split_csv = pd.read_csv(os.path.join(default_configs.HARVARD_DATASET_CONFIG['data_root'],split_type,f'{site}_{split}_split.csv'))
    print('Shape of split_csv:', split_csv.shape)
    
    reference_reports_path = os.path.join(default_configs.HARVARD_DATASET_CONFIG['data_root'],'matched_eeg_recordings_report', site)
    reference_reports_json_files = [os.path.join(reference_reports_path, row['DeidentifiedName(Reports)'].replace('.txt',''),row['DeidentifiedName(Reports)'].replace('.txt','.json')) for _, row in split_csv.iterrows()]
    print('Number of reference reports:', len(reference_reports_json_files))
    print('Reference report path example:', reference_reports_json_files[0])

    results_saved_path = args.results_saved_path
    score_path = os.path.join(results_saved_path, 'scores')
    os.makedirs(score_path, exist_ok=True)
    generated_reports_json_files = glob.glob(os.path.join(results_saved_path, 'generated_reports_json','*.json'))
    print(os.path.join(results_saved_path, 'generated_reports_json'))
    print('Number of generated reports:', len(generated_reports_json_files))
    print('Generated report path example:', generated_reports_json_files[0])

    # Create Log File
    log_file = os.path.join(results_saved_path, f'log_evaluate_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    log_f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    print(f"Log file: {log_file}")

    evaluator = Evaluator(model_name=model_name,ignore_perplexity=ignore_perplexity)

    # initiaize a df to store the scores
    overall_scores_df_columns = ['deidentified_name', 'bleu-1', 'bleu-4', 'bleu-1-smooth', 'bleu-4-smooth', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'perplexity', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'meteor']
    overall_scores_df_rows = []


    for reference_file in reference_reports_json_files:
        deidentified_name = reference_file.split('/')[-1].replace('.json','')
        # check if the deidentified_name is in the generated_reports_json_files
        if os.path.exists(os.path.join(results_saved_path, 'generated_reports_json',f'GENERATED_REPORT_{deidentified_name}.json')):
            generated_file = os.path.join(results_saved_path, 'generated_reports_json',f'GENERATED_REPORT_{deidentified_name}.json')
            
            # read the generated_file
            with open(generated_file, 'r') as f:
                generated_report = json.load(f)
            
            # read the reference_file
            with open(reference_file, 'r') as f:
                reference_report = json.load(f)
            
            # evaluate the generated_report
            score = evaluator.section_wise_metrics(reference_report, generated_report)
            
            overall_scores = evaluator.overall_metrics(score)
            
            final_scores   = {
                "section_wise_scores": score,
                "overall_scores": overall_scores
            }
            
            # add the scores to the df
            overall_scores_df_rows.append({
                'deidentified_name': deidentified_name,
                'bleu-1': overall_scores['bleu_score_results']['bleu-1'],
                'bleu-4': overall_scores['bleu_score_results']['bleu-4'],
                'bleu-1-smooth': overall_scores['bleu_score_results']['bleu-1-smooth'],
                'bleu-4-smooth': overall_scores['bleu_score_results']['bleu-4-smooth'],
                'bertscore_precision': overall_scores['bertscore_results']['precision'],
                'bertscore_recall': overall_scores['bertscore_results']['recall'],
                'bertscore_f1': overall_scores['bertscore_results']['f1'],
                'perplexity': overall_scores['perplexity_results']['perplexity'],
                'rouge1': overall_scores['rouge_score_results']['rouge1'],
                'rouge2': overall_scores['rouge_score_results']['rouge2'],
                'rougeL': overall_scores['rouge_score_results']['rougeL'],
                'rougeLsum': overall_scores['rouge_score_results']['rougeLsum'],
                'meteor': overall_scores['meteor_score_results']['meteor']
            })
            
            
        else:
            print(f'{deidentified_name} is not in the generated_reports_json_files')
            final_scores = {
                "section_wise_scores": None,
                "overall_scores": None
            }
            
            overall_scores_df_rows.append({
                'deidentified_name': deidentified_name,
                'bleu-1': 0.0,
                'bleu-4': 0.0,
                'bleu-1-smooth': 0.0,
                'bleu-4-smooth': 0.0,
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
                'perplexity': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'rougeLsum': 0.0,
                'meteor': 0.0
            })
            
        # save the score
        with open(os.path.join(score_path,f'{deidentified_name}.json'), 'w') as f:
            json.dump(final_scores, f)
                
                
        # break

    overall_scores_df = pd.DataFrame(overall_scores_df_rows, columns=overall_scores_df_columns)
    # reset the index
    overall_scores_df = overall_scores_df.reset_index(drop=True)
    
    print(f'Overall Scores Shape {overall_scores_df.shape}')
    # save the df 
    overall_scores_df.to_csv(os.path.join(results_saved_path, 'overall_scores.csv'), index=False)
