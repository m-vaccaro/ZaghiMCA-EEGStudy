import os
import time
import random
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import pandas as pd


class EEGReadingGUI:
    """
    GUI for EEG reading experiment with single paragraphs and
    a multiple-choice comprehension question after each paragraph.

    Expected CSV columns:
      - REQUIRED: 'text' (paragraph text)
      - OPTIONAL (for questions):
          'question'
          'option_A', 'option_B', 'option_C', 'option_D'
          'correct_option' (e.g., "A", "B", "C", "D")
      - Any additional columns (topic_hint, genre, difficulty, etc.)
        are passed through into the trial log.
    """

    def __init__(self, stimuli_csv_path, shuffle_trials=True):
        # ---- Root window & appearance ----
        self.root = ctk.CTk()
        self.root.title("EEG Reading Experiment")
        self.root.geometry("1280x720")
        ctk.set_appearance_mode("light")

        # ---- Load stimuli ----
        self.stim_df = pd.read_csv(stimuli_csv_path)
        if "text" not in self.stim_df.columns:
            raise ValueError("Stimuli CSV must contain a 'text' column with paragraph text.")

        # Convert each row to a dict so we can easily access fields
        self.paragraphs = self.stim_df.to_dict(orient="records")
        self.n_trials = len(self.paragraphs)

        # Trial order (indices into self.paragraphs / stim_df)
        self.trial_indices = list(range(self.n_trials))
        if shuffle_trials:
            random.shuffle(self.trial_indices)

        # ---- State variables ----
        self.participant_id = None
        self.participant_id_display = None

        # Screens: "participant_id", "instructions", "trial", "question", "end"
        self.current_screen = "participant_id"
        self.current_trial_idx = 0            # index into trial_indices (0..n_trials-1)
        self.current_paragraph_idx = None     # actual row index in stim_df

        self.frame = None                     # current main frame

        # Timing and logging
        self.trial_start_time = None          # reading phase start
        self.question_start_time = None       # question phase start
        self.current_log_entry = None         # dict for current trial
        self.log_rows = []                    # list of log dicts

        # GUI style
        self.button_height = 35
        self.button_width = 200
        self.button_font = ("Open Sans", 20, "bold")
        self.text_font = ("Open Sans", 18)

        # Multiple-choice response variable
        self.mc_response_var = tk.StringVar(value="")

        # Key binding state
        self.space_binding_active = False

    # ---------------------- Top-level control ----------------------

    def run(self):
        """Start the GUI."""
        self.show_participant_id_screen()
        self.root.mainloop()

    def clear_frame(self):
        """Destroy existing frame and create a new one."""
        if self.frame is not None:
            self.frame.destroy()
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

    # ---------------------- Participant ID screen ------------------

    def show_participant_id_screen(self):
        self.current_screen = "participant_id"
        self.clear_frame()

        label = ctk.CTkLabel(
            master=self.frame,
            text="Enter Participant ID:",
            font=("Open Sans", 18, "bold")
        )
        label.pack(pady=(40, 10))

        self.participant_id_entry = ctk.CTkEntry(
            master=self.frame,
            width=200,
            height=30,
            placeholder_text="Participant ID",
            font=("Open Sans", 14),
        )
        self.participant_id_entry.pack(pady=10)

        continue_button = ctk.CTkButton(
            master=self.frame,
            text="Start",
            command=self.on_submit_participant_id,
            height=self.button_height,
            width=self.button_width,
            font=self.button_font,
        )
        continue_button.pack(pady=40)

    def on_submit_participant_id(self):
        pid = self.participant_id_entry.get().strip()
        if not pid:
            messagebox.showinfo("Missing ID", "Please enter a valid Participant ID.")
            return

        confirm = messagebox.askyesno(
            "Confirm ID",
            f"Is this Participant ID correct?\n\n{pid}"
        )
        if not confirm:
            return

        self.participant_id = pid
        self.participant_id_display = f"ID: {pid}"

        os.makedirs("logs", exist_ok=True)

        self.show_instructions_screen()

    # ---------------------- Instructions screen --------------------

    def show_instructions_screen(self):
        self.current_screen = "instructions"
        self.clear_frame()

        header = ctk.CTkLabel(
            master=self.frame,
            text="Welcome!",
            font=("Open Sans", 24, "bold"),
        )
        header.pack(pady=(40, 10))

        instructions = (
            "Thank you for participating in this study.\n\n"
            "In this task, you will read a series of short paragraphs while we record your brain activity.\n\n"
            "For each paragraph:\n"
            "  • Read silently at a comfortable pace.\n"
            "  • When you are done reading, press the SPACEBAR or click the 'Done' button.\n"
            "  • After each paragraph, answer a multiple-choice question about what you read.\n\n"
            "Please try to stay as still as possible during each paragraph to reduce EEG artifacts.\n\n"
            "Press 'Begin' when you are ready to start."
        )

        text_box = ctk.CTkTextbox(
            master=self.frame,
            width=1100,
            height=400,
            wrap="word",
            font=self.text_font,
            corner_radius=0,
            fg_color="transparent",
        )
        text_box.insert("1.0", instructions)
        text_box.configure(state="disabled", cursor="arrow")
        text_box.pack(pady=(10, 20), padx=40)

        begin_button = ctk.CTkButton(
            master=self.frame,
            text="Begin",
            command=self.start_trials,
            height=self.button_height,
            width=self.button_width,
            font=self.button_font,
        )
        begin_button.pack(pady=(10, 40))

        if self.participant_id_display:
            pid_label = ctk.CTkLabel(
                master=self.frame,
                text=self.participant_id_display,
                font=("Open Sans", 12),
            )
            pid_label.pack(padx=20, pady=10, anchor="w")

    # ---------------------- Trial flow control ---------------------

    def start_trials(self):
        self.current_trial_idx = 0
        self.show_trial()

    def bind_space_for_trial(self):
        if not self.space_binding_active:
            self.root.bind("<space>", self.on_space_pressed)
            self.space_binding_active = True

    def unbind_space(self):
        if self.space_binding_active:
            self.root.unbind("<space>")
            self.space_binding_active = False

    def on_space_pressed(self, event=None):
        if self.current_screen == "trial":
            self.finish_reading_phase()

    # ---------------------- Reading screen ------------------------

    def show_trial(self):
        """Show the reading screen for the current trial."""
        if self.current_trial_idx >= self.n_trials:
            self.unbind_space()
            self.show_end_screen()
            return

        self.current_screen = "trial"
        self.clear_frame()
        self.bind_space_for_trial()

        trial_number = self.current_trial_idx + 1
        paragraph_idx = self.trial_indices[self.current_trial_idx]
        self.current_paragraph_idx = paragraph_idx
        row = self.paragraphs[paragraph_idx]
        paragraph_text = row["text"]

        header = ctk.CTkLabel(
            master=self.frame,
            text=f"Paragraph {trial_number} of {self.n_trials}",
            font=("Open Sans", 22, "bold"),
        )
        header.pack(pady=(20, 5))

        instructions = ctk.CTkLabel(
            master=self.frame,
            text="Read silently. Press SPACE or click 'Done' when you are finished.",
            font=("Open Sans", 16),
        )
        instructions.pack(pady=(0, 10))

        text_box = ctk.CTkTextbox(
            master=self.frame,
            width=1100,
            height=450,
            wrap="word",
            font=self.text_font,
            corner_radius=10,
        )
        text_box.insert("1.0", paragraph_text)
        text_box.configure(state="disabled", cursor="arrow")
        text_box.pack(pady=(5, 10), padx=40, fill="both", expand=True)

        done_button = ctk.CTkButton(
            master=self.frame,
            text="Done",
            command=self.finish_reading_phase,
            height=self.button_height,
            width=self.button_width,
            font=self.button_font,
        )
        done_button.pack(pady=(10, 30))

        if self.participant_id_display:
            pid_label = ctk.CTkLabel(
                master=self.frame,
                text=self.participant_id_display,
                font=("Open Sans", 12),
            )
            pid_label.pack(padx=20, pady=10, anchor="w")

        # Start reading timer
        self.trial_start_time = time.time()
        self.current_log_entry = None

    def finish_reading_phase(self):
        """
        Called when participant finishes reading (SPACE or 'Done').
        Logs reading time and then shows the question screen
        for the same paragraph.
        """
        end_time = time.time()
        rt = end_time - (self.trial_start_time or end_time)

        row = self.paragraphs[self.current_paragraph_idx]

        # Initialize log entry for this trial
        log_entry = {
            "participant_id": self.participant_id,
            "trial_index": self.current_trial_idx,
            "stim_index": self.current_paragraph_idx,
            "reading_start_time_unix": self.trial_start_time,
            "reading_end_time_unix": end_time,
            "reading_time_sec": rt,
        }

        # Include all stimulus metadata except text
        for col in self.stim_df.columns:
            if col == "text":
                continue
            log_entry[col] = row.get(col)

        self.current_log_entry = log_entry

        # Move to question page (do NOT increment trial index yet)
        self.unbind_space()
        self.show_question_page()

    # ---------------------- Question screen -----------------------

    def show_question_page(self):
        """Show a multiple-choice question about the current paragraph."""
        self.current_screen = "question"
        self.clear_frame()

        row = self.paragraphs[self.current_paragraph_idx]
        trial_number = self.current_trial_idx + 1

        # Question text from CSV or placeholder
        question_text = row.get(
            "question",
            f"Placeholder question for Paragraph {trial_number}.\n\n"
            f"Please choose one of the options below."
        )

        # Collect options from CSV if present; else placeholders
        options = []
        option_map = [
            ("option_A", "A"),
            ("option_B", "B"),
            ("option_C", "C"),
            ("option_D", "D"),
        ]
        for col_name, label in option_map:
            opt_text = row.get(col_name)
            if opt_text is not None and str(opt_text).strip() != "" and str(opt_text).lower() != "nan":
                options.append((opt_text, label))

        if not options:
            # Placeholder options
            options = [
                ("Placeholder option A", "A"),
                ("Placeholder option B", "B"),
                ("Placeholder option C", "C"),
                ("Placeholder option D", "D"),
            ]

        header = ctk.CTkLabel(
            master=self.frame,
            text=f"Question for Paragraph {trial_number}",
            font=("Open Sans", 22, "bold"),
        )
        header.pack(pady=(20, 10))

        q_label = ctk.CTkLabel(
            master=self.frame,
            text=question_text,
            font=self.text_font,
            wraplength=1000,
            justify="left",
        )
        q_label.pack(pady=(10, 20), padx=40)

        self.mc_response_var.set("")  # clear previous selection

        # Radio buttons for options
        for opt_text, opt_label in options:
            rb = ctk.CTkRadioButton(
                master=self.frame,
                text=f"{opt_label}. {opt_text}",
                value=opt_label,
                variable=self.mc_response_var,
                font=self.text_font,
            )
            rb.pack(anchor="w", padx=60, pady=5)

        continue_button = ctk.CTkButton(
            master=self.frame,
            text="Continue",
            command=self.finish_question_phase,
            height=self.button_height,
            width=self.button_width,
            font=self.button_font,
        )
        continue_button.pack(pady=(20, 30))

        if self.participant_id_display:
            pid_label = ctk.CTkLabel(
                master=self.frame,
                text=self.participant_id_display,
                font=("Open Sans", 12),
            )
            pid_label.pack(padx=20, pady=10, anchor="w")

        # Start question timer
        self.question_start_time = time.time()

    def finish_question_phase(self):
        """Record MC response and move to the next trial."""
        choice = self.mc_response_var.get()
        if choice == "":
            messagebox.showinfo("Response required", "Please select an option before continuing.")
            return

        end_time = time.time()
        q_rt = end_time - (self.question_start_time or end_time)

        if self.current_log_entry is None:
            self.current_log_entry = {}

        self.current_log_entry["question_response"] = choice
        self.current_log_entry["question_start_time_unix"] = self.question_start_time
        self.current_log_entry["question_end_time_unix"] = end_time
        self.current_log_entry["question_time_sec"] = q_rt

        # If CSV has a 'correct_option' column, log correctness
        correct_option = self.paragraphs[self.current_paragraph_idx].get("correct_option")
        if correct_option is not None:
            self.current_log_entry["question_correct_option"] = correct_option
            self.current_log_entry["question_is_correct"] = (choice == correct_option)

        # Append this trial to the log
        self.log_rows.append(self.current_log_entry)
        self.current_log_entry = None

        # Autosave after each trial
        self.save_log_partial()

        # Next trial
        self.current_trial_idx += 1
        self.show_trial()

    # ---------------------- Intermittent saving -------------------
    def save_log_partial(self):
        """Write current log_rows to disk without closing the app."""
        if not self.log_rows:
            return

        if self.participant_id is None:
            fname = "EEG_log_unknown.csv"
        else:
            fname = f"EEG_log_{self.participant_id}.csv"

        os.makedirs("logs", exist_ok=True)
        out_path = os.path.join("logs", fname)

        df_log = pd.DataFrame(self.log_rows)
        df_log.to_csv(out_path, index=False)

    # ---------------------- End screen & saving -------------------

    def show_end_screen(self):
        self.current_screen = "end"
        self.clear_frame()

        title = ctk.CTkLabel(
            master=self.frame,
            text="Thank you!",
            font=("Open Sans", 26, "bold"),
        )
        title.pack(pady=(60, 10))

        msg = (
            "You have completed the reading portion of the study.\n\n"
            "Please let the researcher know you are finished."
        )
        text = ctk.CTkLabel(
            master=self.frame,
            text=msg,
            font=("Open Sans", 18),
        )
        text.pack(pady=(10, 20))

        exit_button = ctk.CTkButton(
            master=self.frame,
            text="Save & Close",
            command=self.save_and_quit,
            height=self.button_height,
            width=self.button_width,
            font=self.button_font,
        )
        exit_button.pack(pady=40)

        if self.participant_id_display:
            pid_label = ctk.CTkLabel(
                master=self.frame,
                text=self.participant_id_display,
                font=("Open Sans", 12),
            )
            pid_label.pack(padx=20, pady=10, anchor="w")

    def save_and_quit(self):
        # Save log_rows to a CSV
        if self.participant_id is None:
            fname = "EEG_log_unknown.csv"
        else:
            fname = f"EEG_log_{self.participant_id}.csv"

        os.makedirs("logs", exist_ok=True)
        out_path = os.path.join("logs", fname)

        df_log = pd.DataFrame(self.log_rows)
        df_log.to_csv(out_path, index=False)
        print(f"Saved trial log to {out_path}")

        self.root.destroy()


if __name__ == "__main__":
    # Point this to your final stimuli CSV
    STIMULI_CSV = "../database_storage/database_18-gpt5_1-full-120_to_150_words.csv"
    app = EEGReadingGUI(STIMULI_CSV, shuffle_trials=True)
    app.run()