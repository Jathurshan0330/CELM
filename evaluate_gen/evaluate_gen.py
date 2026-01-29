import evaluate
from configs.section_mapping import SECTION_STANDARDIZATION_MAPPING
import numpy as np


class Evaluator:
    def __init__(self,model_name,ignore_perplexity=False):
        super().__init__()
        self.model_name = model_name
        self.ignore_perplexity = ignore_perplexity


    def compute_bleu_score(self, references, predictions):
        """
        Compute BLEU-1 BLEU-1-Smooth BLEU-4 BLEU-4-Smooth scores for the predictions.
        """
        
        max_order = [1,4]
        smooth = [False, True]
        bleu_score = evaluate.load("bleu")
        bleu_score_results = {}
        for order in max_order:
            for s in smooth:
                score_name = f'bleu-{order}-smooth' if s else f'bleu-{order}'
                bleu_score_results[score_name] = bleu_score.compute(references=references, predictions=predictions, max_order=order, smooth=s)
        return bleu_score_results
    
    def compute_bertscore(self, references, predictions):
        """
        Compute BERTScore for the predictions. Outputs precision, recall, f1 score.
        """
        bertscore = evaluate.load("bertscore")
        bertscore_results = bertscore.compute(references=references, predictions=predictions, lang="en", model_type="distilbert-base-uncased",device="cpu")
        return bertscore_results
    
    def compute_perplexity(self, predictions):
        """
        Compute perplexity for the predictions.
        """
        perplexity = evaluate.load("perplexity")
        perplexity_results = perplexity.compute(predictions=predictions,model_id=self.model_name) #,device="cpu")
        return perplexity_results
    
    def compute_rouge_score(self, references, predictions):
        """
        Compute ROUGE-1 ROUGE-2 ROUGE-L ROUGE-Lsum scores for the predictions.
        """
        rouge = evaluate.load("rouge")
        rouge_score_results = rouge.compute(references=references, predictions=predictions)
        return rouge_score_results
    
    def compute_meteor_score(self, references, predictions):
        """
        Compute METEOR score for the predictions.
        """
        meteor = evaluate.load("meteor")
        meteor_score_results = meteor.compute(references=references, predictions=predictions)
        return meteor_score_results
    
    def calculate_metrics(self, references, predictions):
        """
        Calculate all the metrics for the predictions.
        """
        try:
            bleu_score_results = self.compute_bleu_score(references, predictions)
        except Exception as e: # error handling
            print(f"Error in bleu_score_results: {e}")
            bleu_score_results = {'bleu-1': {'bleu': 0.0}, 'bleu-4': {'bleu': 0.0}, 'bleu-1-smooth': {'bleu': 0.0}, 'bleu-4-smooth': {'bleu': 0.0}}
        
        try:
            bertscore_results = self.compute_bertscore(references, predictions)
        except Exception as e: # error handling
            print(f"Error in bertscore_results: {e}")
            bertscore_results = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            if not self.ignore_perplexity:
                perplexity_results = self.compute_perplexity(predictions)
            else:
                perplexity_results = {
                    "mean_perplexity": 0.0,
                    "perplexities": 0.0
                }
        except Exception as e: # error handling
            print(f"Error in perplexity_results: {e}")
            perplexity_results = {
                "mean_perplexity": 0.0,
                "perplexities": 0.0
            }
        
        try:
            rouge_score_results = self.compute_rouge_score(references, predictions)
        except Exception as e: # error handling
            print(f"Error in rouge_score_results: {e}")
            rouge_score_results = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}
        
        try:
            meteor_score_results = self.compute_meteor_score(references, predictions)
        except Exception as e: # error handling
            print(f"Error in meteor_score_results: {e}")
            meteor_score_results = {'meteor': 0.0}
        
        return {
            "bleu_score_results": bleu_score_results,
            "bertscore_results": bertscore_results,
            "perplexity_results": perplexity_results,
            "rouge_score_results": rouge_score_results,
            "meteor_score_results": meteor_score_results
        }
        
    def section_wise_metrics(self, reference_report_sections, generated_report_sections):
        """
        Calculate the metrics for each section and all sections to
        """
        section_names = reference_report_sections['extracted_eeg_section_names']
        # Standardize the section names
        section_names = [SECTION_STANDARDIZATION_MAPPING[section_name] for section_name in section_names]
        reference_report_sections['EEG_section_llm_extractions']['EEG_sections'] = [{"section_name":SECTION_STANDARDIZATION_MAPPING[dict_['section_name']], "section_text":dict_['section_text']} for dict_ in reference_report_sections["EEG_section_llm_extractions"]['EEG_sections'] ]
        
        
        section_wise_metrics = {}
        # print(generated_report_sections,len(generated_report_sections['report_sections']))
        try:
            if 'report_sections' in generated_report_sections:
                section_names_in_generated_report = [generated_report_sections['report_sections'][i]['section_name'].lower().strip() for i in range(len(generated_report_sections['report_sections']))]
            else:
                section_names_in_generated_report = []
        
        
                
            for section_name in section_names:
                if section_name.lower().strip() in section_names_in_generated_report:
                    references = [dict_['section_text'] for dict_ in reference_report_sections['EEG_section_llm_extractions']['EEG_sections'] if dict_['section_name'] == section_name]
                    predictions = [dict_['section_text'] for dict_ in generated_report_sections['report_sections'] if dict_['section_name'] == section_name]
                    # print('References:',references)
                    # print('Predictions:',predictions)
                    section_wise_metrics[section_name] = self.calculate_metrics(references, predictions)
                else:
                    section_wise_metrics[section_name] = None
                    
            return section_wise_metrics
        except Exception as e:
            print(f"Error in section_wise_metrics: {e}")
            for section_name in section_names:
                section_wise_metrics[section_name] = None
            return section_wise_metrics
    
    def overall_metrics(self, section_wise_metrics):
        """
        Calculate the overall mean for the section-wise metrics.
        If the section_wise_metrics is None (prompted but not generated), then the metric is 0.0
        """
        try:
            overall_metrics = {
                "bleu_score_results": {
                    "bleu-1": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-1']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-4": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-4']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-1-smooth": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-1-smooth']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-4-smooth": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-4-smooth']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "bertscore_results": {
                    "precision": np.mean([section_wise_metrics[section_name]['bertscore_results']['precision'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "recall": np.mean([section_wise_metrics[section_name]['bertscore_results']['recall'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "f1": np.mean([section_wise_metrics[section_name]['bertscore_results']['f1'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "perplexity_results": {
                    "perplexity": np.mean([section_wise_metrics[section_name]['perplexity_results']['mean_perplexity'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "rouge_score_results": {
                    "rouge1": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rouge1'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rouge2": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rouge2'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rougeL": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rougeL'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rougeLsum": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rougeLsum'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "meteor_score_results": {
                    "meteor": np.mean([section_wise_metrics[section_name]['meteor_score_results']['meteor'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
            }
        except Exception as e:
            print(f"Error in overall_metrics: {e}")
            overall_metrics = {
                "bleu_score_results": {
                    "bleu-1": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-1']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-4": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-4']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-1-smooth": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-1-smooth']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "bleu-4-smooth": np.mean([section_wise_metrics[section_name]['bleu_score_results']['bleu-4-smooth']['bleu'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                       "bertscore_results": {
                    "precision": np.mean([np.mean(section_wise_metrics[section_name]['bertscore_results']['precision']) if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "recall": np.mean([np.mean(section_wise_metrics[section_name]['bertscore_results']['recall']) if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "f1": np.mean([np.mean(section_wise_metrics[section_name]['bertscore_results']['f1']) if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "perplexity_results": {
                    "perplexity": np.mean([section_wise_metrics[section_name]['perplexity_results']['mean_perplexity'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "rouge_score_results": {
                    "rouge1": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rouge1'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rouge2": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rouge2'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rougeL": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rougeL'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                    "rougeLsum": np.mean([section_wise_metrics[section_name]['rouge_score_results']['rougeLsum'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
                "meteor_score_results": {
                    "meteor": np.mean([section_wise_metrics[section_name]['meteor_score_results']['meteor'] if section_wise_metrics[section_name] is not None else 0.0 for section_name in section_wise_metrics.keys() ]),
                },
            }
        return overall_metrics

        
    
