from transformers import pipeline
import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def get_llm_response(pipe,task_prompt: str):
    
    messages = [
        {"role": "user", "content": task_prompt},
    ]
    result = pipe(messages)
    output = result[0]['generated_text'][1]['content']
    return output


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




def normalize_text(text):
    """Normalize text for better comparison."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Convert to lowercase for comparison
    text = text.lower()
    return text

def find_best_match_in_text(section_text, note_text, window_size=None):
    """Find the best matching substring in note_text for section_text."""
    section_norm = normalize_text(section_text)
    note_norm = normalize_text(note_text)
    
    if section_norm in note_norm:
        return section_text, 1.0  # Exact match
    
    # If no exact match, find the closest substring
    # This helps with minor formatting differences
    if window_size is None:
        window_size = len(section_text)
    
    best_match = ""
    best_score = 0.0
    
    # Simple sliding window (for small texts)
    words_note = note_norm.split()
    words_section = section_norm.split()
    
    for i in range(len(words_note) - len(words_section) + 1):
        window = " ".join(words_note[i:i+len(words_section)])
        # Simple word overlap score
        overlap = len(set(words_section) & set(window.split()))
        score = overlap / len(words_section) if words_section else 0
        
        if score > best_score:
            best_score = score
            # Get original text for this window
            best_match = " ".join(note_text.split()[i:i+len(words_section)])
    
    return best_match, best_score

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate text."""
    reference_tokens = normalize_text(reference).split()
    candidate_tokens = normalize_text(candidate).split()
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # Use smoothing to avoid zero scores for short texts
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score (using BLEU-4 by default)
    score = sentence_bleu(
        [reference_tokens], 
        candidate_tokens,
        smoothing_function=smoothing
    )
    
    return score

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores between reference and candidate text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rouge2_p': scores['rouge2'].precision,
        'rouge2_r': scores['rouge2'].recall,
        'rougeL_f': scores['rougeL'].fmeasure,
        'rougeL_p': scores['rougeL'].precision,
        'rougeL_r': scores['rougeL'].recall,
    }

def check_llm_extraction(note_text, sections_json, use_similarity=True):
    """
    Check LLM extraction quality with exact match, BLEU, and ROUGE scores.
    
    Args:
        note_text: Original neurology report text
        sections_json: Extracted sections in JSON format
        use_similarity: If True, also compute similarity scores for non-exact matches
    
    Returns:
        Dictionary with detailed metrics
    """
    sec_name_exact_match = 0
    sec_text_exact_match = 0
    
    total_sections = len(sections_json)
    
    section_metrics = []
    
    for idx, section in enumerate(sections_json):
        section_name = section['section_name']
        section_text = section['section_text']
        
        metrics = {
            'section_index': idx,
            'section_name': section_name,
            'name_exact_match': False,
            'text_exact_match': False,
        }
        
        # Check exact matches
        if section_name in note_text:
            sec_name_exact_match += 1
            metrics['name_exact_match'] = True
        
        if section_text in note_text:
            sec_text_exact_match += 1
            metrics['text_exact_match'] = True
        
        # Calculate similarity scores if requested
        if use_similarity:
            # Find best matching text in the report
            best_match, match_score = find_best_match_in_text(section_text, note_text)
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(section_text, note_text)
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(section_text, best_match)
            
            metrics.update({
                'best_match_text': best_match[:100] + '...' if len(best_match) > 100 else best_match,
                'match_score': match_score,
                'bleu_score': bleu_score,
                **rouge_scores
            })
        
        section_metrics.append(metrics)
    
    # Calculate aggregate metrics
    results = {
        'total_sections': total_sections,
        'name_exact_match_count': sec_name_exact_match,
        'text_exact_match_count': sec_text_exact_match,
        'name_exact_match_rate': sec_name_exact_match / total_sections if total_sections > 0 else 0,
        'text_exact_match_rate': sec_text_exact_match / total_sections if total_sections > 0 else 0,
        'section_details': section_metrics
    }
    
    if use_similarity and section_metrics:
        # Calculate average similarity scores
        avg_bleu = sum(m.get('bleu_score', 0) for m in section_metrics) / total_sections
        avg_rouge1_f = sum(m.get('rouge1_f', 0) for m in section_metrics) / total_sections
        avg_rouge2_f = sum(m.get('rouge2_f', 0) for m in section_metrics) / total_sections
        avg_rougeL_f = sum(m.get('rougeL_f', 0) for m in section_metrics) / total_sections
        
        results.update({
            'avg_bleu_score': avg_bleu,
            'avg_rouge1_f': avg_rouge1_f,
            'avg_rouge2_f': avg_rouge2_f,
            'avg_rougeL_f': avg_rougeL_f,
        })
    
    return results
