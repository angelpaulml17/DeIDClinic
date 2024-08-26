import io
import ttkbootstrap as ttk
import warnings
from ttkbootstrap.tooltip import ToolTip
from idlelib.redirector import WidgetRedirector


class ReadOnlyText(ttk.Text):
    """A ttk.Text widget but the user cannot edit it, only select and copy etc."""

    def __init__(self, *args, **kwargs):
        ttk.Text.__init__(self, *args, **kwargs)
        self.redirector = WidgetRedirector(self)
        self.insert = self.redirector.register('insert', lambda *a, **kw: 'break')
        self.delete = self.redirector.register('delete', lambda *a, **kw: 'break')


class MaskTagTooltip(ToolTip):
    """A subclass of ttkbootstrap's ToolTip class, but it shows the tooltip when the user hovers over one of the
     given tags in a Text widget"""

    def __init__(self, *args, **kwargs):
        # Function used to determine which replacement has been clicked
        self.detect_function = kwargs.pop('detect_function', MaskTagTooltip._noop)
        # Function used to decide what text is displayed when the tooltip is hovered
        self.text_function = kwargs.pop('text_function', MaskTagTooltip.default_text)
        # Function used to handle what happens when the tag is clicked
        self.click_function = kwargs.pop('click_function', MaskTagTooltip._noop)
        # List of tags to bind the events to
        self.tags = kwargs.pop('tags', [])
        # List of replacements (MaskOperations) associated with the document
        self.replacements = kwargs.pop('replacements', [])
        ToolTip.__init__(self, *args, **kwargs)
        for tag in self.tags:
            self.widget.tag_bind(tag, '<Enter>', self.show_tooltip)
            self.widget.tag_bind(tag, '<Leave>', self.hide_tooltip)
            self.widget.tag_bind(tag, '<1>', self.click_tag)

    @staticmethod
    def default_text(replacement):
        return replacement.old_token

    @staticmethod
    def _noop(*args, **kwargs):
        return False

    # Skip default tooltip behaviour, we don't want it to show when just mousing over the widget.
    def enter(self, event=None):
        pass

    def hide_tooltip(self, event):
        self.widget.config(cursor='arrow')
        self.hide_tip()

    def show_tooltip(self, event):
        self.widget.config(cursor='hand2')
        replacement = self.get_replacement(event.x, event.y)
        if replacement:
            self.text = self.text_function(replacement)
            self.show_tip()

    def click_tag(self, event):
        replacement = self.get_replacement(event.x, event.y)
        if replacement:
            self.click_function(replacement)

    def get_replacement(self, x, y):
        char = self.widget.count('1.0', f"@{x},{y}", 'chars')[0]
        for r in self.replacements:
            if self.detect_function(r, char):
                return r

        return None


class TextWidgetLogger(io.StringIO):
    """IO-like object for streaming text directly to a tk.Text widget"""

    def __init__(self, widget):
        self.widget = widget
        # Get initial contents of widget so it is in sync with the internal StringIO, just in case we ever
        # want to use any of the other StringIO methods to read it.
        super().__init__(self.widget.get('1.0', 'end-1c'))

    def write(self, text):
        self.widget.insert(ttk.INSERT, text)
        self.widget.see(ttk.END)  # Scroll widget to end
        return super().write(text)


class MultiselectBox(ttk.Menubutton):
    def __init__(self, *args, **kwargs):
        self.options = kwargs.pop('options', [])
        self.max_choices = kwargs.pop('max_choices', 1)
        ttk.Menubutton.__init__(self, *args, **kwargs)
        self.menu = ttk.Menu(self, tearoff=False)
        self.configure(menu=self.menu)

        self.choices = {}
        for opt in self.options:
            self.choices[opt] = ttk.IntVar(value=0)
            self.menu.add_checkbutton(label=opt, variable=self.choices[opt],
                                      onvalue=1, offvalue=0,
                                      command=self.handle_select)

        self.update_text([])

    # Disable/enable checkboxes to ensure "max_choices" is enforced
    def handle_select(self):
        selected = self.get_selected()
        state = ttk.NORMAL
        if len(selected) > self.max_choices:
            state = ttk.DISABLED
        last = self.menu.index(ttk.END)
        for i in range(last + 1):
            self.menu.entryconfigure(i, state=state)
            # Make sure options that are checked are not disabled, otherwise they can never be unchecked.
            if self.choices[self.menu.entrycget(i, 'label')].get():
                self.menu.entryconfigure(i, state=ttk.NORMAL)
        self.update_text(selected)

    def update_text(self, selected):
        if len(selected) > 0:
            self['text'] = ', '.join(selected)
        else:
            self['text'] = '-'

    def get_selected(self):
        return [name for name, var in self.choices.items() if var.get() == 1]

    def set_selected(self, selection):
        for selected in selection:
            choice = self.choices.get(selected)
            if choice:
                choice.set(1)
            else:
                warnings.warn(f"'{selected}' is not a recognized option")
        self.handle_select()
