from .settings import *
import tkinter as tk
from tkinter import ttk, scrolledtext
import pywinstyles, sys
import sv_ttk
import queue

class App(tk.Tk):
    def __init__(self, subtitle):
        super().__init__()
        self.subtitle = subtitle
        self.title(title + ' ' + subtitle)
        self.geometry(f"{width}x{height}")
        sv_ttk.set_theme(theme=theme)
        self.apply_theme_to_titlebar()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style(self)
        style.configure("CustomStatus.TLabel", background=self["background"])

        self.queue = queue.Queue()
        self.status_label = ttk.Label(self, text="Idle", style="CustomStatus.TLabel")
        self.status_label.pack(pady=10)

        style = ttk.Style()
        style.configure("Tree.TFrame", background=self["background"])

        self.last_opened_parent = None  # Track last opened parent

        # --- Frame to hold Treeview and Scrollbar ---
        self.tree_frame = ttk.Frame(self, style="Tree.TFrame")
        self.tree_frame.pack(expand=True, fill="both", pady=10, padx=10)

        # --- Scrollbar ---
        self.scrollbar = ttk.Scrollbar(self.tree_frame)
        self.scrollbar.pack(side="right", fill="both")

        # --- Treeview ---
        self.treeview = ttk.Treeview(
            self.tree_frame,
            selectmode="browse",
            yscrollcommand=self.scrollbar.set,
            columns=(),
            show="tree",
            height=15
        )
        self.treeview.pack(expand=True, fill="both")
        self.scrollbar.config(command=self.treeview.yview)

        self.start_button = ttk.Button(self, text="Start", command=lambda: None, width=15, style="Accent.TButton")
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(self, text="Stop", command=lambda: None, width=15, style="Accent.TButton")
        self.stop_button.pack(pady=5)

        self.item_id_counter = 1  # unique IDs for inserting items
        self.check_queue()

    def apply_theme_to_titlebar(self):
        version = sys.getwindowsversion()
        if version.major == 10 and version.build >= 22000:
            pywinstyles.change_header_color(self, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")
        elif version.major == 10:
            pywinstyles.apply_style(self, "dark" if sv_ttk.get_theme() == "dark" else "normal")
            self.wm_attributes("-alpha", 0.99)
            self.wm_attributes("-alpha", 1)

    def check_queue(self):
        while not self.queue.empty():
            result = self.queue.get()
            self.insert_treeview_data(result)
        self.after(100, self.check_queue)

    def insert_treeview_data(self, data):
        text, sentiment = data  # Unpack tuple

        parent_id = str(self.item_id_counter)
        self.treeview.insert("", "end", iid=parent_id, text=f"🗣 {text}")
        self.item_id_counter += 1

        child_id = str(self.item_id_counter)
        self.treeview.insert(parent_id, "end", iid=child_id, text=f"→ Sentiment: {sentiment}")
        self.item_id_counter += 1

        # Close the last opened parent if it exists
        if self.last_opened_parent:
            self.treeview.item(self.last_opened_parent, open=False)

        # Open the new parent and scroll to child
        self.treeview.item(parent_id, open=True)
        self.treeview.see(child_id)

        # Update the tracker
        self.last_opened_parent = parent_id

    def set_start_command(self, func):
        self.start_button.config(command=func)

    def set_stop_command(self, func):
        self.stop_button.config(command=func)

    def set_status(self, text):
        self.status_label.config(text=text)

    def on_closing(self):
        self.destroy()
