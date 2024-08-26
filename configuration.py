import importlib
import yaml


class Configuration:
    """Class for holding Mask configuration"""
    def __init__(self, **kwargs):
        self.filename = None

        self.project_name = None
        self.project_start_date = None
        self.project_owner = None
        self.project_owner_contact = None

        self.dataset_location = None
        self.data_output = None
        self.ehost_output = None
        self.csv_output = None

        self.entity = {}

        self.instances = {}
        self.algorithms = []
        self.properties = []  # Keep track of all the different properties, used when saving back to the config file.
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.properties.append(key)
        self.set_algorithms()

    def load(self, filename='configuration.yml'):
        """Loading function that can take configuration file, or it uses default location:
            configuration.yml file in folder where mask_framework is"""
        self.filename = filename
        conf_doc = open(self.filename).read()
        conf_doc = yaml.safe_load(conf_doc)
        for key, value in conf_doc.items():
            setattr(self, key, value)
            self.properties.append(key)
        self.set_algorithms()

    def save(self, filename=None):
        if filename is None:
            filename = self.filename or 'configuration.yml'
        to_dump = {}
        for p in self.properties:
            to_dump[p] = getattr(self, p)

        # Need to convert algorithms back to config format.
        entity_config = {}
        for alg_settings in self.algorithms:
            entity_name = alg_settings.pop('entity_name')
            # Set up a dict with the entity name as the key, or get the existing dict
            settings = entity_config.get(entity_name, {})
            algs = settings.get('algorithm', [])  # Get the list of algorithms, or set up a new one
            algs.append(alg_settings.pop('algorithm'))  # Add the algorithm to the list
            settings['algorithm'] = algs
            settings.update(alg_settings)  # Merge other properties ("masking_class" etc.)
            entity_config[entity_name] = settings

        to_dump['entity'] = entity_config

        for entity_name in to_dump['entity']:
            entity_config = to_dump['entity'][entity_name]
            if isinstance(entity_config['algorithm'], list):
                alg_dict = {}
                for i, alg in enumerate(entity_config['algorithm']):
                    alg_dict[f"algorithm{i+1}"] = alg

                entity_config['algorithm'] = alg_dict

        with open(filename, 'w') as config_file:
            yaml.dump(to_dump, config_file, indent=2)

    def load_ner_plugin(self, algorithm):
        """Instantiate the given NER plugin class name, unless already loaded"""
        if algorithm not in self.instances:
            self.instances[algorithm] = getattr(importlib.import_module('ner_plugins.' + algorithm), algorithm)()

    def load_masking_plugin(self, algorithm):
        """Instantiate the given masking plugin class name, unless already loaded"""
        if algorithm not in self.instances:
            self.instances[algorithm] = getattr(importlib.import_module('masking_plugins.' + algorithm), algorithm)()

    def set_algorithms(self):
        """Set an array of algorithms based on the provided entity config"""
        self.algorithms = []
        for entity_name in self.entity:
            entity_config = self.entity[entity_name]

            if isinstance(entity_config['algorithm'], str):
                entity_config['algorithm'] = [entity_config['algorithm']]
            elif isinstance(entity_config['algorithm'], dict):
                entity_config['algorithm'] = list(entity_config['algorithm'].values())

            for algorithm in entity_config['algorithm']:
                self.algorithms.append({'entity_name': entity_name,
                                        'algorithm': algorithm,
                                        'resolution': entity_config.get('resolution'),
                                        'masking_type': entity_config['masking_type'],
                                        'masking_class': entity_config['masking_class']})

    def instantiate(self):
        """Instantiate one of each NER and masking plugin class"""
        unused = set(self.instances.keys())
        for algorithm_settings in self.algorithms:
            self.load_ner_plugin(algorithm_settings['algorithm'])
            unused.discard(algorithm_settings['algorithm'])
            self.load_masking_plugin(algorithm_settings['masking_class'])
            unused.discard(algorithm_settings['masking_class'])

        # Remove any unused instances
        for key in unused:
            self.instances.pop(key, None)
