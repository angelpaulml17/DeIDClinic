import os
import queue
import threading
import time
import logging
import tkinter as tk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs.dialogs import Messagebox

import mask_framework as mask_framework
from configuration import Configuration
from mask_output import MaskOutput
from gui.config import Config as MaskGuiConfig
from gui.utils import *


class MaskThread(threading.Thread):
    """Thread to handle to the running of the mask framework execution. Notifies `status_queue` when finished."""

    def __init__(self, status_queue, mask_gui):
        super().__init__()
        self.queue = status_queue
        self.mask_gui = mask_gui

    def _mask(self):
        mask_framework.execute(self.mask_gui.config, self.mask_gui.current_run)

    def _test(self):
        time.sleep(5)

    def run(self):
        try:
            self._mask()
            self.queue.put('Done')
        except:
            self.queue.put('Error')
            logging.error("Fatal error when masking", exc_info=True)


class MaskGui(ttk.Frame):
    """The main Mask GUI application window"""

    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.config = Configuration()
        self.config.load()

        # MAIN CONTAINER
        self.tabs = ttk.Notebook(self.parent)
        self.tabs.grid(row=0, column=0, sticky='nesw')

        # Page 1 - Inputs and settings
        self.input_page = ttk.Frame(self.tabs)
        self.input_page.grid_rowconfigure(0, weight=1)
        self.input_page.grid_columnconfigure(0, weight=1)
        self.input_page.grid_columnconfigure(1, weight=1)
        self.input_page.grid(row=0, column=0, sticky='nesw')

        self.input_frame = ttk.Frame(self.input_page)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.input_frame.grid(row=0, column=0, padx=15, pady=5, sticky='nesw')

        self.algorithm_frame = ttk.Frame(self.input_page)
        self.algorithm_frame.grid_columnconfigure(1, weight=0)
        self.algorithm_frame.grid_columnconfigure(2, weight=3)
        self.algorithm_frame.grid_columnconfigure(3, weight=1)
        self.algorithm_frame.grid_columnconfigure(4, weight=3)
        self.algorithm_frame.grid(row=0, column=1, padx=15, pady=5, sticky='nesw')

        # Page 2 - Run status and output
        self.run_page = ttk.Frame(self.tabs)
        self.run_page.grid_rowconfigure(0, weight=0)
        self.run_page.grid_rowconfigure(1, weight=0)
        self.run_page.grid_rowconfigure(2, weight=1)
        self.run_page.grid_columnconfigure(0, weight=1)
        self.run_page.grid(row=0, column=0, sticky='nesw')

        self.run_frame = ttk.Frame(self.run_page)
        self.run_frame.grid_rowconfigure(0, weight=1)
        self.run_frame.grid_columnconfigure(1, weight=1)
        self.run_frame.grid(row=1, column=0, padx=5, pady=15, sticky='nesw')

        self.output_frame = ttk.Frame(self.run_page)
        self.output_frame.grid_rowconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid(row=2, column=0, padx=5, pady=5, sticky='nesw')

        self.tabs.add(self.input_page, text='Home')
        self.tabs.add(self.run_page, text='Run', state='hidden')

        # INPUT
        row_offset = 0

        # Store a dict of each configuration key and the UI element containing its value
        self.config_inputs = {}

        # Misc run metadata
        for setting in ['project_name', 'project_start_date', 'project_owner', 'project_owner_contact']:
            setting_label = ttk.Label(self.input_frame, text=f"{setting.replace('_', ' ').capitalize()}")
            setting_entry = ttk.Entry(self.input_frame)
            setting_entry.insert(0, getattr(self.config, setting))
            setting_label.grid(row=row_offset, column=0, padx=5, pady=5, sticky='ne')
            setting_entry.grid(row=row_offset, column=1, padx=5, pady=5, sticky='nesw')
            self.config_inputs[setting] = setting_entry
            row_offset += 1

        # Input/output directory selection
        def build_dir_select_function(entry):
            def handle_select_dir():
                directory = tk.filedialog.askdirectory(
                    title='Select a directory',
                    initialdir=entry.get())
                if directory:
                    entry.delete(0, ttk.END)
                    entry.insert(0, directory)

            return handle_select_dir

        for setting in ['dataset_location', 'data_output']:
            setting_label = ttk.Label(self.input_frame, text=f"{setting.replace('_', ' ').capitalize()}")
            setting_entry = ttk.Entry(self.input_frame)
            setting_entry.insert(0, os.path.abspath(getattr(self.config, setting)))
            setting_label.grid(row=row_offset, column=0, padx=5, pady=5, sticky='ne')
            setting_entry.grid(row=row_offset, column=1, padx=5, pady=5, sticky='nesw')
            dir_open_button = ttk.Button(
                self.input_frame,
                text='Browse',
                command=build_dir_select_function(setting_entry),
                bootstyle=SECONDARY
            )
            dir_open_button.grid(row=row_offset, column=2, padx=5, pady=5, sticky='w')
            self.config_inputs[setting] = setting_entry
            row_offset += 1

        # ALGORITHMS
        self.config_algorithm_inputs = {}

        # Display a table-like view of each entity type
        # Headings
        for pos, label in enumerate([None, 'Type', 'Algorithms', 'Resolution', 'Masking Type']):
            if label:
                ttk.Label(self.algorithm_frame, text=label).grid(row=0, column=pos, padx=5, pady=5, sticky='nw')

        row_offset = 1
        for setting in MaskGuiConfig.entity_options:
            entity = setting['key']
            setting_label = ttk.Label(self.algorithm_frame, text=setting['label'])

            enabled = entity in self.config.entity
            setting_checkbox_var = tk.IntVar(value=1 if enabled else 0)
            setting_checkbox = ttk.Checkbutton(self.algorithm_frame, variable=setting_checkbox_var)

            # NER options/algorithms
            available_ner_options = [MaskGuiConfig.ner_options_inverse[k] for k in setting['ner_options']]
            setting_ner_combobox = MultiselectBox(self.algorithm_frame, options=available_ner_options, bootstyle='secondary')
            # Set the initial selection from the provided config file...
            ner_opts = self.config.entity[entity]['algorithm'] if enabled else []
            selected_ner_options = [MaskGuiConfig.ner_options_inverse[k] for k in ner_opts]
            setting_ner_combobox.set_selected(selected_ner_options)

            # Resolution options
            max_str_length = max(len(x) for x in MaskGuiConfig.resolution_options)
            setting_resolution_combobox = ttk.Combobox(self.algorithm_frame, width=max_str_length)
            setting_resolution_combobox['values'] = MaskGuiConfig.resolution_options
            setting_resolution_combobox['state'] = 'readonly'
            res_opt = self.config.entity[entity].get('resolution') if enabled else MaskGuiConfig.resolution_options[0]
            try:
                res_idx = setting_resolution_combobox['values'].index(res_opt)
            except ValueError:
                res_idx = 0
            setting_resolution_combobox.current(res_idx)

            # Mask options
            setting_mask_combobox = ttk.Combobox(self.algorithm_frame)
            setting_mask_combobox['values'] = [MaskGuiConfig.masking_options_inverse[k] for k in
                                               setting['masking_options']]
            setting_mask_combobox['state'] = 'readonly'
            selected_mask_class = self.config.entity[entity]['masking_class'] if enabled else None
            try:
                mask_opt = MaskGuiConfig.masking_options_inverse.get(selected_mask_class)
                mask_idx = setting_mask_combobox['values'].index(mask_opt)
            except ValueError:
                mask_idx = 0
            setting_mask_combobox.current(mask_idx)

            setting_checkbox.grid(row=row_offset, column=0, padx=5, pady=5, sticky='ne')
            setting_label.grid(row=row_offset, column=1, padx=5, pady=5, sticky='nw')
            setting_ner_combobox.grid(row=row_offset, column=2, padx=5, pady=5, sticky='nesw')
            setting_resolution_combobox.grid(row=row_offset, column=3, padx=5, pady=5, sticky='nesw')
            setting_mask_combobox.grid(row=row_offset, column=4, padx=5, pady=5, sticky='nesw')

            row_offset += 1

            self.config_algorithm_inputs[entity] = {
                'checkbox': setting_checkbox_var,
                'ner_setting': setting_ner_combobox,
                'mask_setting': setting_mask_combobox,
                'resolution_setting': setting_resolution_combobox
            }

        ttk.Separator(self.input_page).grid(row=2, column=0, columnspan=2, padx=15, pady=5, sticky='ew')

        # Save config button
        self.save_config_button = ttk.Button(
            self.input_page,
            text='Save Settings',
            command=self.handle_save_config,
            bootstyle='secondary'
        )
        self.save_config_button.grid(row=3, column=0, padx=15, pady=5, sticky='sw', ipadx=20, ipady=10)

        # Run button
        self.run_button = ttk.Button(
            self.input_page,
            text='Run',
            command=self.handle_run
        )
        self.run_button.grid(row=3, column=1, padx=15, pady=5, sticky='se', ipadx=20, ipady=10)

        ## RUN
        # Status labels
        self.input_status = ttk.Label(self.run_frame, text='')
        self.algorithm_status = ttk.Label(self.run_frame, text='')

        # Progress bars
        self.input_progress_bar = ttk.Progressbar(self.run_frame, orient='horizontal')
        self.algorithm_progress_bar = ttk.Progressbar(self.run_frame, orient='horizontal')
        self.input_status.grid(row=1, column=1, padx=5, pady=5)
        self.input_progress_bar.grid(row=2, column=1, padx=5, pady=5, sticky='nesw')
        self.algorithm_status.grid(row=3, column=1, padx=5, pady=5)
        self.algorithm_progress_bar.grid(row=4, column=1, padx=5, pady=5, sticky='nesw')

        # OUTPUT
        # Output tabs
        self.output_tabs = ttk.Notebook(self.output_frame)
        self.output_tabs.grid(row=0, column=0, sticky='nesw')

        # Mask log
        self.log_frame = ttk.Frame(self.output_tabs)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid(row=0, column=0, sticky='nesw')
        self.log = ReadOnlyText(self.log_frame)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, orient=ttk.VERTICAL, command=self.log.yview)
        self.log['yscrollcommand'] = self.log_scrollbar.set
        self.log.grid(row=0, column=0, padx=(10, 0), pady=10, sticky='nesw')
        self.log_scrollbar.grid(row=0, column=1, pady=10, sticky='ns')

        # Results list
        self.results_frame = ttk.Frame(self.output_tabs)
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=0)
        self.results_frame.grid_columnconfigure(2, weight=3)
        self.results_frame.grid(row=0, column=0, sticky='nesw')
        self.results_list = tk.Listbox(self.results_frame, selectmode=ttk.SINGLE)
        self.results_list.bind('<<ListboxSelect>>', self.handle_result_select)
        self.results_list_scrollbar = ttk.Scrollbar(self.results_frame, orient=ttk.VERTICAL, command=self.log.yview)
        self.results_list['yscrollcommand'] = self.results_list_scrollbar.set

        #  Previews
        self.previews = ttk.Notebook(self.results_frame)
        #    Input
        self.input_preview_frame = ttk.Frame(self.previews)
        self.input_preview_frame.grid_rowconfigure(0, weight=1)
        self.input_preview_frame.grid_columnconfigure(0, weight=1)
        self.input_preview_frame.grid(row=0, column=0, sticky='nesw')
        self.input_preview = ReadOnlyText(self.input_preview_frame)
        self.input_preview_scrollbar = ttk.Scrollbar(self.input_preview_frame, orient=ttk.VERTICAL,
                                                     command=self.input_preview.yview)
        self.input_preview['yscrollcommand'] = self.input_preview_scrollbar.set
        self.input_preview.grid(row=0, column=0, padx=(10, 0), pady=10, sticky='nesw')
        self.input_preview_scrollbar.grid(row=0, column=1, pady=10, sticky='ns')

        #    Output
        self.output_preview_frame = ttk.Frame(self.previews)
        self.output_preview_frame.grid_rowconfigure(0, weight=1)
        self.output_preview_frame.grid_columnconfigure(0, weight=1)
        self.output_preview_frame.grid(row=0, column=0, sticky='nesw')
        self.output_preview = ReadOnlyText(self.output_preview_frame)

        for tag, colour in MaskGuiConfig.tags.items():
            self.input_preview.tag_config(tag, background=colour)
            self.output_preview.tag_config(tag, background=colour)

        self.output_preview_scrollbar = ttk.Scrollbar(self.output_preview_frame, orient=ttk.VERTICAL,
                                                      command=self.output_preview.yview)
        self.output_preview['yscrollcommand'] = self.output_preview_scrollbar.set
        self.output_preview.grid(row=0, column=0, padx=(10, 0), pady=10, sticky='nesw')
        self.output_preview_scrollbar.grid(row=0, column=1, pady=10, sticky='ns')

        self.previews.add(self.input_preview_frame, text='Input')
        self.previews.add(self.output_preview_frame, text='Output')

        self.results_list.grid(row=0, column=0, padx=(10, 0), pady=10, sticky='nesw')
        self.results_list_scrollbar.grid(row=0, column=1, pady=10, sticky='ns')

        self.output_tabs.add(self.results_frame, text='Results')
        self.output_tabs.add(self.log_frame, text='Log')

        self.show_input_page()

        self.current_run = None

    def show_input_page(self):
        self.tabs.select(0)

    def show_run_page(self):
        self.tabs.tab(1, state="normal")
        self.tabs.select(1)

    def set_configuration(self):
        """Update the internal Configuration object using the values from the fields in the UI.
            Performs validation and returns boolean declaring validity."""

        # Read from Entry fields
        for key in self.config_inputs:
            value = self.config_inputs[key].get()
            setattr(self.config, key, value)

        # Store CSVs in same place as data
        self.config.csv_output = self.config.data_output
        if not self.config.dataset_location or len(self.config.dataset_location) < 1:
            Messagebox.show_error(
                title='Error',
                message='No dataset selected!',
                parent=self.run_frame
            )

            return False

        # Algorithm settings
        algorithms = []
        for key, fields in self.config_algorithm_inputs.items():
            if fields['checkbox'].get() == 1:
                selected_algorithms = [MaskGuiConfig.ner_options[a] for a in fields['ner_setting'].get_selected()]
                for algorithm in selected_algorithms:
                    algorithms.append({'entity_name': key,
                                       'algorithm': algorithm,
                                       'resolution': fields['resolution_setting'].get(),
                                       # TODO: Allow setting this - or just add redaction as a masking algorithm?
                                       'masking_type': 'Mask',
                                       'masking_class': MaskGuiConfig.masking_options[fields['mask_setting'].get()]})

        self.config.algorithms = algorithms

        return True

    def handle_save_config(self):
        if self.set_configuration():
            self.config.save()
            Messagebox.show_info(
                title='Notice',
                message='Settings saved successfully',
                parent=self.input_page
            )

    def handle_run(self):
        self.show_run_page()
        self.input_status['text'] = "Loading configuration"

        if self.set_configuration():
            # Clear previous run
            self.results_list.delete(0, ttk.END)
            self.log.delete('1.0', ttk.END)
            self.clear_previews()

            # Set run button state
            self.run_button['text'] = "Running..."
            self.run_button['state'] = ttk.DISABLED
            self.reset_progress_bars()
            self.input_status['text'] = "Loading algorithms..."
            self.config.instantiate()
            log = TextWidgetLogger(self.log)

            def check():
                try:
                    mask_output = status_queue.get_nowait()
                    if mask_output == 'Error':
                        Messagebox.show_error(
                            title='Error',
                            message='An unexpected error occurred while masking.',
                            parent=self.input_frame
                        )
                    else:
                        self.render_output()

                    self.run_button['state'] = ttk.NORMAL
                    self.run_button['text'] = "Run"
                    mask_thread.join()
                    self.finish_progress_bars()
                except queue.Empty:
                    self.parent.after(100, check)

            status_queue = queue.Queue()
            self.current_run = MaskOutput(self.config.project_name, log, self)
            mask_thread = MaskThread(status_queue, self)
            mask_thread.start()
            self.log.insert(ttk.INSERT, 'Loading...\n')
            self.parent.after(100, check)

    def reset_progress_bars(self):
        self.input_progress_bar['value'] = 0
        self.algorithm_progress_bar['value'] = 0
        self.input_status['text'] = ""
        self.algorithm_status['text'] = ""

    def finish_progress_bars(self):
        self.input_status['text'] = "Finished"
        self.algorithm_status['text'] = ""
        self.input_progress_bar['value'] = 100
        self.algorithm_progress_bar['value'] = 100

    def handle_result_select(self, event):
        selection = self.results_list.curselection()
        if selection and self.current_run:
            f = self.current_run.files[selection[0]]
            self.render_file(f)

    def render_output(self):
        if len(self.current_run.files) > 0 and not self.results_list.curselection():
            self.results_list.select_set(0)
            self.results_list.event_generate('<<ListboxSelect>>')

    def clear_previews(self):
        self.input_preview.delete('1.0', ttk.END)
        self.output_preview.delete('1.0', ttk.END)
        self.previews.grid_forget()

    def render_file(self, masked_file):
        self.clear_previews()
        self.previews.grid(row=0, column=2, padx=(10, 0), pady=10, sticky='nesw')
        self.input_preview.insert(ttk.INSERT, open(masked_file.input_path, 'r').read())
        self.output_preview.insert(ttk.INSERT, open(masked_file.output_path, 'r').read())

        # Show "Input" tab by default
        self.previews.select(0)

        # Tooltips showing mask info when a tag is hovered

        def jump_to_new_token(rep):  # Switch to "Output" tab and move cursor to the replaced token (and scroll)
            self.previews.select(1)
            position = f"1.0+{rep.new_start_index}c"
            self.output_preview.mark_set(tk.INSERT, position)
            self.output_preview.see(position)

        MaskTagTooltip(self.input_preview,
                       text_function=lambda rep: f"{rep.new_token} ({rep.entity} - {rep.algorithm})",
                       detect_function=lambda rep, char: rep.old_start_index <= char < rep.old_end_index,
                       click_function=jump_to_new_token,
                       tags=MaskGuiConfig.tags,
                       replacements=masked_file.replacements)

        def jump_to_old_token(rep):
            self.previews.select(0)
            position = f"1.0+{rep.old_start_index}c"
            self.input_preview.mark_set(tk.INSERT, position)
            self.input_preview.see(position)

        MaskTagTooltip(self.output_preview,
                       text_function=lambda rep: f"{rep.old_token} ({rep.entity} - {rep.algorithm})",
                       detect_function=lambda rep, char: rep.new_start_index <= char < rep.new_end_index,
                       click_function=jump_to_old_token,
                       tags=MaskGuiConfig.tags,
                       replacements=masked_file.replacements)

        for r in masked_file.replacements:
            tag = r.entity
            if r.conflict_token:
                tag = 'Conflict'
            self.input_preview.tag_add(tag, f"1.0+{r.old_start_index}c", f"1.0+{r.old_end_index}c")
            self.output_preview.tag_add(tag, f"1.0+{r.new_start_index}c", f"1.0+{r.new_end_index}c")


if __name__ == '__main__':
    icon_file = os.path.join(os.path.dirname(__file__), 'mask_icon.png')
    root = ttk.Window(themename='flatly', iconphoto=icon_file)
    root.title('Mask')
    root.resizable(True, True)
    root.geometry('1200x600')
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    MaskGui(root).grid(row=0, column=0, sticky='nesw')
    root.mainloop()
