import os
from datetime import datetime


class MaskOutput:
    """Class to represent the output of a mask framework run on a dataset."""

    def __init__(self, project_name, log_file, mask_gui=None):
        self.project_name = project_name
        self.start_time = datetime.now()
        self.end_time = None
        self.files = []
        self.log_file = log_file
        self.input_current = 1
        self.input_max = 1
        self.algorithm_current = 1
        self.algorithm_max = 1
        self.mask_gui = mask_gui

    def set_current_input(self, i):
        self.input_current = i
        if self.mask_gui:
            self.mask_gui.input_progress_bar['value'] = ((i - 0.5) / self.input_max) * 100

    def set_current_algorithm(self, i):
        self.algorithm_current = i
        if self.mask_gui:
            self.mask_gui.algorithm_progress_bar['value'] = ((i - 0.5) / self.algorithm_max) * 100

    def set_input_status(self, status):
        if self.mask_gui:
            self.mask_gui.input_status['text'] = f"{status} ({self.input_current}/{self.input_max})"

    def set_algorithm_status(self, status):
        if self.mask_gui:
            self.mask_gui.algorithm_status['text'] = f"{status} ({self.algorithm_current}/{self.algorithm_max})"

    def log(self, msg):
        self.log_file.write(msg)

    def mask(self, input_path, output_path, csv_path):
        file = MaskedFile(self, input_path, output_path, csv_path)
        self.files.append(file)
        return file

    def begin(self):
        self.log("Project name: " + self.project_name + "\n")
        self.log("Time of run: " + str(self.start_time) + "\n\n")
        self.log("RUN LOG \n")

    def finish(self):
        self.end_time = datetime.now()


class MaskedFile:
    """Class to represent the output of a mask framework run on a single file."""

    def __init__(self, mask_output, input_path, output_path, csv_path):
        self.mask_output = mask_output  # Parent MaskOutput object
        self.input_path = input_path
        self.output_path = output_path
        self.csv_path = csv_path
        self.replacements = []

    def replace(self, algorithm, entity, masking_type, old_token, new_token, old_start_index, old_end_index,
                new_start_index=-1, new_end_index=-1, conflict_token=None):
        op = MaskOperation(self, algorithm, entity, masking_type, old_token, new_token, old_start_index, old_end_index,
                           new_start_index, new_end_index, conflict_token)
        self.replacements.append(op)
        # op.log_replace()
        return op

    def begin(self):
        self.log("Running stats for file: " + self.filename() + '\n')
        if self.mask_output.mask_gui:
            self.mask_output.set_input_status(f"Masking {self.filename()}")

    def finish(self):
        self.log('END for file:' + self.filename() + '\n')
        self.log('========================================================================')
        self.sort_replacements()
        if self.mask_output.mask_gui:
            self.mask_output.mask_gui.results_list.insert(self.mask_output.input_current,
                                                          f"{self.filename()} - {len(self.replacements)} Replacements")

    def sort_replacements(self):
        def by_start_index(row):
            return row.old_start_index
        self.replacements.sort(key=by_start_index)

        # For each replacement, recalculate the new token's position in the output file
        drift = 0
        for r in self.replacements:
            r.new_start_index = r.old_start_index + drift
            r.new_end_index = r.old_start_index + drift + len(r.new_token)
            drift += (len(r.new_token) - len(r.old_token))

    def filename(self):
        return os.path.basename(self.input_path)

    def log(self, msg):
        self.mask_output.log(msg)


class MaskOperation:
    """Class to represent a single token replacement in a file."""

    def __init__(self, masked_file, algorithm, entity, masking_type, old_token, new_token, old_start_index,
                 old_end_index, new_start_index, new_end_index, conflict_token):
        self.masked_file = masked_file  # Parent MaskedFile object
        self.algorithm = algorithm  # The name of the algorithm that performed the masking
        self.entity = entity  # Which entity type was masked
        self.masking_type = masking_type  # Redact/Mask
        self.old_token = old_token  # The original token
        self.new_token = new_token  # The replacement token
        self.old_start_index = old_start_index  # Start index of the replacement the original file
        self.old_end_index = old_end_index  # End index of the replacement in the original file
        self.new_start_index = new_start_index  # Start index of the replacement the modified file
        self.new_end_index = new_end_index  # End index of the replacement in the modified file
        self.conflict_token = conflict_token  # The conflicting replacement token of the second algorithm

    def log(self, msg):
        self.masked_file.log(msg)

    def log_replace(self):
        self.log(self.describe() + '\n')

    def describe(self):
        return f"{self.masking_type.upper()}ED ENTITY: {self.entity} with span " \
               f"({self.old_start_index}, {self.old_end_index})" \
               f" -- {self.old_token}->{self.new_token} --" \
               f" ({self.new_start_index}, {self.new_end_index})"
