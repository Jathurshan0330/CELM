SECTION_STANDARDIZATION_MAPPING = {
    'details:': 'EEG DESCRIPTION/DETAILS',
    'detail:': 'EEG DESCRIPTION/DETAILS',
    'description:': 'EEG DESCRIPTION/DETAILS',
    
    'impression:': 'IMPRESSION/INTERPRETATION',
    'interpretation:': 'IMPRESSION/INTERPRETATION',
    
    'background:': 'BACKGROUND ACTIVITY',
    'background activity:': 'BACKGROUND ACTIVITY',
    
    
    'seizures:' : 'SEIZURES',
    'events/seizures:': 'EVENTS/SEIZURES',
    
    'epileptiform abnormalities:': 'EPLEPTIFORM ABNORMALITIES',
    'interictal epileptiform abnormalities:': 'INTERICTAL EPLEPTIFORM ABNORMALITIES',
    'sleep:': 'SLEEP',
}

STANDARDIZED_SECTION_DESCRIPTIONS = {
    'EEG DESCRIPTION/DETAILS': 'Detailed narrative of EEG findings including background activity, sleep stages, physiologic variants, and abnormalities observed during the recording period.',
    'IMPRESSION/INTERPRETATION': 'Clinical summary and interpretation of findings, including overall assessment (normal/abnormal), significance of abnormalities, and correlation with clinical context.',
    'EVENTS/SEIZURES': 'Documentation of any events captured during recording, including both clinical seizures and non-epileptic events (button presses, symptoms, behavioral changes), with their time of occurrence, description, and corresponding EEG patterns.',
    'SEIZURES':  'Specific documentation of confirmed seizure activity, including electrographic and/or clinical seizures, with details on onset time, duration, semiology, ictal EEG patterns, and post-ictal changes.',
    'BACKGROUND ACTIVITY': 'Description of the dominant posterior rhythm, organization, symmetry, reactivity, and overall continuity of baseline cerebral activity.',
    'EPLEPTIFORM ABNORMALITIES': 'Description of epileptiform discharges including spikes, sharp waves, spike-wave complexes, their location, frequency, and distribution.',
    'INTERICTAL EPLEPTIFORM ABNORMALITIES': 'Epileptiform discharges occurring between seizures, including morphology, lateralization, field of distribution, and activation by procedures (hyperventilation, photic stimulation).',
    'SLEEP': 'Description of sleep architecture, sleep stages achieved, sleep spindles, vertex waves, K-complexes, and any sleep-related changes in epileptiform activity.',
}