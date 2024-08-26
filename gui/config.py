class Config:
    # NER algorithm options to be offered in the UI
    # Key is the text that will be shown to the user, and the value is the class name as in configuration.yml
    ner_options = {'MedCAT': 'MedCAT',
                   'NER CRF': 'NER_CRF',
                   'NER CRF Dictionaries': 'NER_CRF_dictionaries',
                   'NER BERT': 'NER_BERT'}
    # As above but with key/values inverted for fast lookups
    ner_options_inverse = {v: k for k, v in ner_options.items()}

    # Mask algorithm options to be offered in the UI
    # Key/value mapping same as above
    masking_options = {
        'Replace with "NAME"': 'mask_name_simple',
        'Replace with "LOCATION"': 'mask_location_simple',
        'Replace with "DATE"': 'Mask_date_simple',
        'Replace with "ID"': 'mask_id_simple',
        'Replace with "AGE"': 'mask_age_simple',
        'Replace with "CONTACT"': 'mask_contact_simple',
        'Replace with a random job title': 'mask_job_randomized'
    }
    masking_options_inverse = {v: k for k, v in masking_options.items()}

    max_ner_choices = 2 # Number of NER algorithms that can be selected per entity type
    # Options that are shared with all entities.
    shared_ner_options = ['MedCAT', 'NER_CRF', 'NER_CRF_dictionaries', 'NER_BERT']
    # Dict to control which options should appear for each entity type
    entity_options = [
        {'key': 'NAME', 'label': 'Name', 'ner_options': shared_ner_options, 'masking_options': ['mask_name_simple']},
        {'key': 'DATE', 'label': 'Date', 'ner_options': shared_ner_options, 'masking_options': ['Mask_date_simple']},
        {'key': 'LOCATION', 'label': 'Location', 'ner_options': shared_ner_options,
         'masking_options': ['mask_location_simple']},
        {'key': 'CONTACT', 'label': 'Contact', 'ner_options': shared_ner_options,
         'masking_options': ['mask_contact_simple']},
        {'key': 'AGE', 'label': 'Age', 'ner_options': shared_ner_options, 'masking_options': ['mask_age_simple']},
        {'key': 'ID', 'label': 'ID', 'ner_options': shared_ner_options, 'masking_options': ['mask_id_simple']}
    ]

    resolution_options = ['union', 'intersection']

    # The "tags" that will be used to highlight replaced terms in the input/output previews.
    # The keys are the tag names and the values are the colours to highlight them.
    # Colour reference: https://www.tcl.tk/man/tcl/TkCmd/colors.html
    tags = {
        # General
        'Mask': 'pale green',  # Unused
        'Redact': 'yellow',  # Unused
        'Conflict': 'salmon1',  # Unused?

        # Entity types
        'NAME': 'light blue',
        'DATE': 'DarkSeaGreen1',
        'LOCATION': 'lavender',
        'CONTACT': 'thistle2',
        'AGE': 'MistyRose2',
        'ID': 'pale green',
        'PHI': 'yellow',  # Unused?
        'PROFESSION': 'bisque2',  # Unused?
    }
