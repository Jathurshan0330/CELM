UNIMODAL_TEXT_ONLY_BASELINE_PROMPT_WITH_PATIENT_HISTORY = """
You are an expert clinical neurophysiologist specializing in EEG interpretation and clinical report generation.

**TASK**
Your task is to generate the specified sections of a formal clinical EEG report using only the provided:
- Patient history
- EEG description

**EEG SECTION DESCRIPTIONS**
[STANDARDIZED_SECTION_DESCRIPTIONS]

**GUIDELINES**
- Generate only the sections listed in **SECTIONS TO BE GENERATED**.
- Do NOT generate any additional sections.
- Do NOT repeat the same section more than once.
- Only generate the output in the JSON format and do not include any other text or explanation.

**OUTPUT FORMAT (STRICT)** 
Return ONLY the following JSON structure, with no preamble, explanation, or markdown:
```json
{"report_sections": [
    {"section_name": "Name of the section as given in SECTIONS TO BE GENERATED",
     "section_text": "Generated text for the section in string"},
    ...
]}
```

**SECTIONS TO BE GENERATED**
[SECTION_NAMES]

**PATIENT HISTORY AND EEG DESCRIPTION**s
[PATIENT_HISTORY_AND_EEG_DESCRIPTION]

Now generate the EEG report. 
"""


UNIMODAL_TEXT_AND_EEG_STATISTICS_BASELINE_PROMPT = """
**EEG-DERIVED STATISTICS**
[EEG_DERIVED_STATISTICS]

**EEG CHANNELS**
[EEG_CHANNELS]

You are an expert clinical neurophysiologist specializing in EEG interpretation and clinical report generation.

**TASK**
Your task is to generate the specified sections of a formal clinical EEG report using only the provided:
- Patient history
- EEG description
- EEG Channels
- EEG-derived statistics (provided above)

**EEG SECTION DESCRIPTIONS**
[STANDARDIZED_SECTION_DESCRIPTIONS]

**GUIDELINES**
- Generate only the sections listed in **SECTIONS TO BE GENERATED**.
- Do NOT generate any additional sections.
- Do NOT repeat the same section more than once.
- Only generate the output in the JSON format and do not include any other text or explanation.

**OUTPUT FORMAT (STRICT)** 
Return ONLY the following JSON structure, with no preamble, explanation, or markdown:
```json
{"report_sections": [
    {"section_name": "Name of the section as given in SECTIONS TO BE GENERATED",
     "section_text": "Generated text for the section in string"},
    ...
]}
```

**SECTIONS TO BE GENERATED**
[SECTION_NAMES]

**PATIENT HISTORY AND EEG DESCRIPTION**
[PATIENT_HISTORY_AND_EEG_DESCRIPTION]

Now generate the EEG report. 
"""



EEG_LLM_PROJECTION_ONLY_PROMPT = """
**EEG CHANNELS**
[EEG_CHANNELS]

You are an expert clinical neurophysiologist specializing in EEG interpretation and clinical report generation.

**TASK**
Your task is to generate the specified sections (**SECTIONS TO BE GENERATED**) of a formal clinical EEG report using the above provided data of EEG recording sessions and following information:
- Patient history
- EEG description
- EEG Channels

**EEG SECTION DESCRIPTIONS**
[STANDARDIZED_SECTION_DESCRIPTIONS]

**GUIDELINES**
- Generate only the sections listed in **SECTIONS TO BE GENERATED**.
- Do NOT generate any additional sections.
- Do NOT repeat the same section more than once.
- Only generate the output in the JSON format and do not include any other text or explanation.

**OUTPUT FORMAT (STRICT)** 
Return ONLY the following JSON structure, with no preamble, explanation, or markdown:
```json
{"report_sections": [
    {"section_name": "Name of the section as given in SECTIONS TO BE GENERATED",
     "section_text": "Generated text for the section in string"},
    ...
]}
```

**SECTIONS TO BE GENERATED**
[SECTION_NAMES]

**PATIENT HISTORY AND EEG DESCRIPTION**
[PATIENT_HISTORY_AND_EEG_DESCRIPTION]

Now generate the EEG report. 
"""

TUEV_PROMPT = """
You are an expert clinical neurophysiologist specializing in EEG analysis and report generation. Given the EEG recordings session above classify the EEG into one of the following categories:

**Event Types**
- spike and slow wave (spsw)
- generalized periodic epileptiform discharge (gped)
- periodic lateralized epileptiform dischage (pled)
- eye movement (eyem)
- artifact (artf)
- background (bckg)

**Output Format**
Return ONLY the following JSON structure, with no preamble, explanation, or markdown:
```json
{"label": "Event Type (e.g. spsw, gped, pled, eyem, artf, bckg)"}
"""



ABNORMAL_PREDICTION_PROMPT = """
**EEG CHANNELS**
[EEG_CHANNELS]

You are an expert clinical neurophysiologist specializing in EEG analysis. 

**TASK**
Given the EEG signal above and following information, classify the EEG into one of the following categories:
- abnormal
- normal

**PATIENT HISTORY AND EEG DESCRIPTION**
[PATIENT_HISTORY_AND_EEG_DESCRIPTION]

**OUTPUT FORMAT**
Return ONLY the following JSON structure, with no preamble, explanation, or markdown:
```json
{"prediction": "abnormal" or "normal"}
```

Now analyze the signal and predict the label.
"""


UNIMODAL_TEXT_AND_EEG_STATISTICS_ABNORMAL_PREDICTION_PROMPT = """
**EEG-DERIVED STATISTICS**
[EEG_DERIVED_STATISTICS]

**EEG CHANNELS**
[EEG_CHANNELS]

You are an expert clinical neurophysiologist specializing in EEG analysis. 

**TASK**
Given the EEG signal above and following information, classify the EEG into one of the following categories:
- abnormal
- normal

**PATIENT HISTORY AND EEG DESCRIPTION**
[PATIENT_HISTORY_AND_EEG_DESCRIPTION]

**OUTPUT FORMAT**
Return ONLY the following JSON structure, with no preamble, explanation, or additional text:
```json
{"prediction": "abnormal" or "normal"}
```

Now analyze the signal and predict the label.
"""

class PromptGenerator:
    def __init__(self, task: str):
        self.task = task

    def get_prompt(self):
        if self.task == 'unimodal_text_only_baseline':
            return UNIMODAL_TEXT_ONLY_BASELINE_PROMPT_WITH_PATIENT_HISTORY
        elif self.task == 'unimodal_text_and_eeg_features_baseline':
            return UNIMODAL_TEXT_AND_EEG_STATISTICS_BASELINE_PROMPT
        elif self.task == 'eeg_llm_projection_only':
            return EEG_LLM_PROJECTION_ONLY_PROMPT
        elif self.task == 'tuev':
            return TUEV_PROMPT
        elif self.task == 'abnormal_prediction':
            return ABNORMAL_PREDICTION_PROMPT  
        elif self.task == 'unimodal_text_and_eeg_features_abnormal_prediction':
            return UNIMODAL_TEXT_AND_EEG_STATISTICS_ABNORMAL_PREDICTION_PROMPT