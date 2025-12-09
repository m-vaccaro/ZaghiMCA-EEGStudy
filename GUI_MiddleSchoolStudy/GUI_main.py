import itertools
import threading
import time
import random
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import pandas as pd
from PIL import Image
from customtkinter import CTkImage

import openai_interact_rewrite
import openai_interact_profile


class GUI:
    """A GUI for the MCA project, using the customtkinter library"""

    def __init__(self):
        """Initialize the GUI, doesn't start running it.
        Pages:
         page 0: Ask researchers to input participant ID
         page 1: welcome page with participant instructions
         page 2: pre-study survey questions
         page 3: instruction 2.0
         page 4: page where users can interact with GPT-4
         (need a small but discrete button we can use to advance to next page, students should not press the button.
         Can it request a 4-digit PIN and won't continue without it?)
         page 5: post-study survey questions
         page 6: the close page.

        Magic methods defined:
         .len()
         """

        self.participant_id_entry = None
        self.experiment_version_entry = None  # Initialize an empty string to identify Experimental or Control
        self.participant_id = None
        self.experiment_version = None
        self.participant_id_display = None
        self.root = ctk.CTk()
        self.root.title("Learning Style Survey")
        self.root.geometry("1280x720")
        # self.root.configure(bg="white")
        ctk.set_appearance_mode("light")
        # ctk.set_default_color_theme("colors.json")

        # Declare variables for FYI
        self.text = {}  # will initialize later to make this code readable
        self.current_page = 0
        self.runtimes = {}
        self.user_choices = {}
        self.notUser_choices = {}

        self.startTime = 0  # this will be reset in each create_page()
        self.endTime = 0
        self.calcTime = 0
        # self.frame  - this will be created and destroyed in each create_page()

        self.startGUI = 0
        self.quitGUI = 0

        # Define the number of GPT interactions to complete
        self.num_gpt_interactions = 2  # Maximum depends on the length of "original_texts" in openai_interact_rewrite.
        #                                If exceeded, an error will occur.
        self.user_selections = []
        self.user_selections_text = []
        self.user_notSelections = []
        self.user_notSelections_text = []

        # Initialize the text
        self.init_text()

        # Ensure the contents are centered and evenly spaced.
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create an attribute to check if an option has been selected.
        self.option_selected = False

        # Define global options
        self.button_height = 35
        self.button_width = 200
        self.button_font = ("Open Sans", 20, "bold")

        self.option_font = ("Open Sans", 18, "bold")  # radio buttons

        self.text_font = ("Open Sans", 18)  # text boxes
        self.text_box_radius = 10

        self.refresh_button_font = ("Open Sans", 10)
        self.refresh_button_fg_color = '#BCBCBC'
        self.refresh_button_hover = '#808080'

        # Create an attribute to track clicked text boxes. Will use to handle the highlight tracking.
        self.clicked_text_box = None

        # Create the paragraph pair list for the first four paragraphs
        self.paragraph_pair_list = []

        # A variable to keep track of student profile
        self.student_profile = None
        self.student_profile_opposite = None

        # Closing page track selection
        self.closing_selection = None

        # Image paths for loaded images
        self.image_paths = ['Icons/WaterCycle.png', 'Icons/ClimateChange.png', 'Icons/Photosynthesis.png',
                            'Icons/StatesOfMatter.png', 'Icons/PlateTectonics.png', 'Icons/Electricity.png']

        # GPT-paragraph variables
        self.generated_text_iter1 = None
        self.generated_text_iter2 = None

        self.gpt_paragraph1_iter1 = ""
        self.gpt_paragraph2_iter1 = ""
        self.gpt_paragraph1_iter2 = ""
        self.gpt_paragraph2_iter2 = ""

        self.gpt_refresh_code = -1

    def create_participant_id_screen(self):
        participantIDfont = ("Open Sans", 14)
        '''Create the initial participant ID input screen'''
        # Create a new frame
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Ask for Participant ID
        label = ctk.CTkLabel(master=self.frame,
                             text="Enter Participant ID and select Group:",
                             font=participantIDfont)
        label.pack(pady=20)

        # Input box for ID
        self.participant_id_entry = ctk.CTkEntry(master=self.frame, font=participantIDfont,
                                                 width=200, height=25,
                                                 placeholder_text="Enter Participant ID Here")

        self.participant_id_entry.pack(pady=10)

        # Dropdown menu for Experimental or Control
        self.experiment_version_entry = ctk.StringVar(value="Select Group")  # Default value
        group_options = ["Experimental", "Control"]
        group_dropdown = ctk.CTkOptionMenu(master=self.frame, variable=self.experiment_version_entry,
                                           width=200, height=25,
                                           values=group_options, font=participantIDfont)
        group_dropdown.pack(pady=20)

        # Button to submit and continue
        continue_button = ctk.CTkButton(master=self.frame, text="Submit",
                                        command=self.on_submit_participant_id,
                                        height=self.button_height,
                                        width=self.button_width,
                                        font=self.button_font)
        continue_button.pack(pady=(629, 0))

    def on_submit_participant_id(self):
        '''Handle the submission of the Participant ID'''
        participant_id = self.participant_id_entry.get().strip()
        selected_group = self.experiment_version_entry.get()

        if participant_id and selected_group not in ["", "Select Group"]:  # Make sure that the ID is not empty
            confirmation = messagebox.askyesno("Confirm Details",
                                               f"Are the Participant ID '{participant_id}' "
                                               f"and Group '{selected_group}' correct?")
            if confirmation:
                # If the user clicks 'Yes', store the ID and group and continue.
                self.participant_id = participant_id
                self.experiment_version = selected_group
                self.participant_id_display = "ID: " + participant_id

                # Write the participant ID to the output file.
                filename = f"{time.strftime('%Y-%m-%d %H-%M-%S')}" + f" ID-{participant_id}" + ".txt"
                with open(filename, 'w') as f:
                    f.write(f"Participant ID: {self.participant_id}\n\n")

                # Change the stdout to the file to capture further output.
                sys.stdout = open(filename, 'a')

                # Initialize the file & table:
                print(f"Group: {self.experiment_version}")
                print("\nGUI Start time: " + str(time.time()) + " [Unix epoch, in sec.]")
                print("\nGUI User Data (For Tabular Items)\n" +
                      "Page   Button            Time (sec) since page started \n" +
                      "------------------------------------------------------")

                # Destroy the current frame and move on
                self.frame.destroy()
                self.create_welcome()
            else:
                # If the user clicks 'No', let them re-enter the participant ID.
                self.participant_id_entry.delete(0, tk.END)
        else:
            # Error message for empty Participant ID or Group:
            error_label = ctk.CTkLabel(master=self.frame,
                                       text="Please input a valid Participant ID and/or select a Group.")
            error_label.pack(pady=10)

            # Set a timeout for the error message
            self.root.after(5000, error_label.destroy)  # value is a time in milliseconds

    def init_text(self):
        """Initialize the text for the pages from files"""
        # Get the welcome page text
        self.text[0] = ["Welcome", ]
        with open("instructions.txt", 'r') as file:
            self.text[0].append(file.read())

        # Get the paragraph selection page text
        df = pd.read_csv("input3.csv")
        for i, series in df.iterrows():
            self.text[i] = [series['PageTitle'], series['Paragraph1'],
                            series['Paragraph2']]  # Shift index by 1 for initial paragraphs
            # print(i, self.text[i])

        # Starting index for GPT pages is after the last initial paragraph
        gpt_start_index = len(self.text)

        # Add GPT interaction pages
        for i in range(self.num_gpt_interactions):
            gpt_page_num = gpt_start_index + i  # Calculate GPT page number
            self.text[gpt_page_num] = [f"GPT Interaction {i + 1}", "Placeholder text 1", "Placeholder text 2"]

        # Set the close page text
        self.text[len(self.text)] = ["Close", "Thank you for your participation! \nEnter any other feedback you have "
                                              "here:", ""]

    def run(self):
        '''Run the CTk mainloop() and our first page'''
        self.current_page = 0
        self.runtimes[self.current_page] = []
        self.create_participant_id_screen()
        self.root.mainloop()

    def create_welcome(self):
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Create a header frame to hold and display instructions/title
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        # Pack in the instructions/title
        ctkTitle = ctk.CTkLabel(master=header_frame, text="Welcome!", font=('Open Sans', 24, 'bold'))
        ctkTitle.pack(padx=30, pady=(15, 5), anchor='n')

        ctkTitle2 = ctk.CTkLabel(master=header_frame,
                                 text="Thank you for taking the time to complete this activity!",
                                 font=('Open Sans', 18), height=10, wraplength=1200)

        # Add an image
        ctk_image = CTkImage(light_image=Image.open("Icons/welcome.png"), size=(500, 500))
        image_label = ctk.CTkLabel(master=header_frame, image=ctk_image, text='')
        image_label.pack(padx=0, pady=10)
        ctkTitle2.pack(padx=30, pady=0, anchor='s')

        # Button to submit and continue
        continue_button = ctk.CTkButton(master=self.frame, text="Continue",
                                        command=self.create_instructions,
                                        height=self.button_height,
                                        width=self.button_width,
                                        font=self.button_font)
        continue_button.pack(pady=(0, 40), anchor='s')

        # Display participant ID
        participantID = ctk.CTkLabel(master=self.frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=85, pady=10, side='bottom', anchor='w')

        # Start timer on welcome page.
        self.startTime = time.time()

    def create_instructions(self):
        # Calculate time the welcome page was open and destroy the welcome page.
        self.endTime = time.time()
        self.calcTime = self.endTime - self.startTime
        self.startGUI = self.endTime
        print(f"Welc   Continue           " +
              f"{self.calcTime:.2f} s")
        self.frame.destroy()

        # An option selection is not required for the welcome page.
        self.option_selected = True

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        instructions = ("Thank you for taking the time to complete this activity -- your participation is helping to "
                        "revolutionize the future of learning!\n\nOn the following few screens, you will see two "
                        "explanations of the same science topic, which you may or may not have some background "
                        "knowledge on. The explanations will be similar, with slight differences between them. "
                        "You will be asked to select the explanation you prefer. There is no right answer, so go with "
                        "your gut! You will see a total of six pairs.\n\nAt the "
                        "end of the activity, you will see two possible descriptions of your learning preferences. "
                        "From there, you will be asked to select the one you think describes you best and to provide "
                        "a brief justification for your choice."
                        "\n\nWhen you are ready to begin, please select \"Accept & Continue\" below.")

        # Create a header frame to hold and display instructions/title
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        # Pack in the instructions/title
        ctkTitle = ctk.CTkLabel(master=header_frame, text="Directions", font=('Open Sans', 24, 'bold'))
        ctkTitle.pack(padx=30, pady=(15, 5), anchor='n')

        ctkText = ctk.CTkTextbox(master=header_frame, width=1240, height=490, wrap="word", corner_radius=0,
                                 fg_color='transparent', font=self.text_font)

        ctkText.insert('1.0', instructions)
        ctkText.configure(state='disabled', cursor="arrow")
        ctkText.pack(pady=(10, 5), padx=30)

        # Button to continue
        accept_button = ctk.CTkButton(self.frame, text="Accept & Continue", command=self.click_next_page,
                                      height=self.button_height,
                                      width=self.button_width,
                                      font=self.button_font)
        accept_button.pack(pady=(240, 0))

        # Display participant ID
        participantID = ctk.CTkLabel(master=self.frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=85, pady=10, side='bottom', anchor='w')

        # Start the timer
        self.startTime = time.time()

    def on_enter(self, event, text_box):
        if self.clicked_text_box != text_box:
            text_box.configure(border_color="light blue", border_width=5)

    def on_leave(self, event, text_box):
        if self.clicked_text_box != text_box:
            text_box.configure(border_color="#DBDBDB", border_width=0)

    def on_click(self, event, text_box, radio_var):
        # Unhighlight the previously clicked textbox if it exists
        if self.clicked_text_box and self.clicked_text_box != text_box:
            self.clicked_text_box.configure(border_color="#DBDBDB", border_width=0)

        # Update the clicked textbox and its appearance
        self.clicked_text_box = text_box
        text_box.configure(border_color="light blue", border_width=5)

        selected_value = radio_var.get()

        normal_color = "transparent"
        selected_color = "light blue"
        if selected_value == 1:
            # Highlight left frame and unhighlight right frame.
            self.left_option_frame.configure(fg_color=selected_color)
            self.right_option_frame.configure(fg_color=normal_color)
        elif selected_value == 2:
            # Highlight right frame and unhighlight left frame.
            self.left_option_frame.configure(fg_color=normal_color)
            self.right_option_frame.configure(fg_color=selected_color)

        # When the user clicks a radio button, save their choice and print it
        self.runtimes[self.current_page].append(time.time() - self.startTime)
        self.user_choices[self.current_page] = radio_var.get()
        if radio_var.get() == 1:  # Save the opposite of the user choice as well
            self.notUser_choices[self.current_page] = 2  # If they selected the left option, save the right (not chosen)
        else:
            self.notUser_choices[self.current_page] = 1  # If they selected the right option, save the left one.
        print(f"pg {self.current_page}   Option Select: {radio_var.get()}   " +
              f"{self.runtimes[self.current_page][-1]:.2f} s")
        # Update "option_selected" to reflect that one of the two options has been selected.
        self.option_selected = True

    def create_paragraph_page(self, title, paragraph1, paragraph2, radio_var, gpt_interaction_num):
        self.paragraph_pair_list.append(f"{title} \n\nParagraph 1:\n{paragraph1}\n\nParagraph 2:\n{paragraph2}")

        # Page setup
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)
        self.clicked_text_box = None  # Reset tracking parameter to None each time a new page is created.

        '''Create a page with two paragraph options and radio buttons.'''

        # Create a header frame to hold title and image
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', pady=(20, 0), padx=50)

        # Load and resize the image
        ctk_image = CTkImage(light_image=Image.open(self.image_paths[self.current_page - 1]), size=(350, 350))

        # Create an image label
        image_label = ctk.CTkLabel(master=header_frame, image=ctk_image, text='')

        # Page title
        ctkTitle = ctk.CTkLabel(master=header_frame, text=title, font=("Open Sans", 24, "bold"), height=10)

        # Pack into frame
        ctkTitle.pack(padx=0, pady=(0, 10), side='top', anchor='center')
        image_label.pack(padx=0)

        # Page instructions
        selectInstructions = ("Please select your preferred paragraph from the two options below."
                              "\n(Remember there is no correct response.)")
        ctkInstructions = ctk.CTkLabel(master=self.frame, text=selectInstructions, font=("Open Sans", 20))
        ctkInstructions.pack(pady=(5, 0))

        # Set up radio button variable and trace changes to it
        radio_var = tk.IntVar(value=0)

        # Create a divided frame for the text options
        options_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        options_frame.pack(fill='both', expand=True, padx=60, pady=(10, 5))

        # Creating left option frame
        self.left_option_frame = ctk.CTkFrame(options_frame, fg_color="transparent", corner_radius=15)
        self.left_option_frame.pack(side='left', fill='both', expand=True, padx=(10, 15))

        # Left Text label
        option_1_label = ctk.CTkLabel(master=self.left_option_frame, text="Option 1", font=self.option_font)
        option_1_label.pack(padx=(0, 0), pady=(5, 0), anchor='n')

        # Left Text option
        self.left_text = ctk.CTkTextbox(self.left_option_frame, font=self.text_font,
                                        corner_radius=self.text_box_radius, wrap="word", border_width=0,
                                        border_color="#DBDBDB", cursor='hand2')
        self.left_text.insert(tk.END, paragraph1)
        self.left_text.configure(state='disabled')  # Disable textbox after all text is inserted
        self.left_text.bind("<Button-1>", lambda event: radio_var.set(1))
        self.left_text.pack(padx=(25, 25), pady=(5, 25), fill='both', expand=True)
        self.left_text.bind("<Enter>", lambda event, t=self.left_text: self.on_enter(event, t))
        self.left_text.bind("<Leave>", lambda event, t=self.left_text: self.on_leave(event, t))
        self.left_text.bind("<Button-1>", lambda event, t=self.left_text: self.on_click(event, t, radio_var))

        # Creating right option frame
        self.right_option_frame = ctk.CTkFrame(options_frame, fg_color="transparent", corner_radius=15)
        self.right_option_frame.pack(side='right', fill='both', expand=True, padx=(15, 10))

        # Right Text label
        option_2_label = ctk.CTkLabel(master=self.right_option_frame, text="Option 2", font=self.option_font)
        option_2_label.pack(padx=(0, 0), pady=(5, 0), anchor='n')

        # Right Text option
        self.right_text = ctk.CTkTextbox(self.right_option_frame, font=self.text_font,
                                         corner_radius=self.text_box_radius, wrap="word", border_width=0,
                                         border_color="#DBDBDB", cursor='hand2')
        self.right_text.insert(tk.END, paragraph2)
        self.right_text.configure(state='disabled')  # Disable textbox after all text is inserted
        self.right_text.bind("<Button-1>", lambda event: radio_var.set(2))
        self.right_text.pack(padx=(25, 25), pady=(5, 25), fill='both', expand=True)
        self.right_text.bind("<Enter>", lambda event, t=self.right_text: self.on_enter(event, t))
        self.right_text.bind("<Leave>", lambda event, t=self.right_text: self.on_leave(event, t))
        self.right_text.bind("<Button-1>", lambda event, t=self.right_text: self.on_click(event, t, radio_var))

        # Continue button
        continue_button_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        continue_button_frame.pack(fill='x', side='top', padx=30, pady=(0, 10))

        continue_button = ctk.CTkButton(continue_button_frame, text="Continue",
                                        command=lambda: self.click_next_page(radio_var.get()),
                                        height=self.button_height,
                                        width=self.button_width,
                                        font=self.button_font)
        continue_button.pack(pady=20)

        # Handle the refresh button creation - always create the frame for consistency, but only add in the button if
        # it is a GPT page.

        # Add a refresh button to the bottom right corner of the button frame
        refresh_button_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        refresh_button_frame.pack(fill='x', side='top', padx=60, pady=0)

        # Display Participant ID
        participantID = ctk.CTkLabel(master=refresh_button_frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))

        if gpt_interaction_num == 0:
            refresh_button = ctk.CTkButton(refresh_button_frame, text='',
                                           height=20,
                                           width=30,
                                           fg_color='transparent',
                                           state='disabled')
            refresh_button.pack(padx=25, pady=10, side='right')
            participantID.pack(padx=25, pady=10, side='left')

        if gpt_interaction_num > 0:
            refresh_button = ctk.CTkButton(refresh_button_frame, text="Refresh",
                                           command=lambda: self.click_refresh(gpt_interaction_num),
                                           height=20,
                                           width=30,
                                           font=self.refresh_button_font,
                                           fg_color=self.refresh_button_fg_color,
                                           hover_color=self.refresh_button_hover)
            refresh_button.pack(padx=25, pady=10, side='right')
            participantID.pack(padx=25, pady=10, side='left')

        # Start the timer
        self.startTime = time.time()

    def create_page(self, interaction=None):
        """Create the current page based on the type of page (regular or GPT interaction)"""
        # Each time a new page is created, ensure the option_selected flag is set to False.
        self.option_selected = False

        # Destroy the previous frame if it exists
        if hasattr(self, 'frame'):
            self.frame.destroy()

        # Check if the current page is a "regular page" (i.e., from CSV/a base survey paragraph)
        if self.current_page < len(self.text) - self.num_gpt_interactions - 1:
            title, paragraph1, paragraph2 = self.text[self.current_page]
            radio_var = tk.IntVar(value=0)
            self.create_paragraph_page(title, paragraph1, paragraph2, radio_var, 0)

        # Check if the current page is a GPT interaction page
        elif self.current_page in range(len(self.text) - self.num_gpt_interactions - 1, len(self.text) - 1):
            if interaction is None:
                # GPT interaction page setup
                if self.generated_text_iter1 is None:  # Then it is the first GPT interaction page
                    self.create_gpt_interaction_page()
                # Otherwise profiles and texts have already been generated, so we can move on w/o "loading"
                if self.generated_text_iter1 is not None:
                    self.startTime = time.time()
                    self.update_gpt_interaction_page(self.gpt_paragraph1_iter2, self.gpt_paragraph2_iter2)
            if interaction is not None:
                self.gpt_refresh_code = interaction
                self.create_gpt_interaction_page()

        # If none of the above, it's the closing page
        else:
            self.break_screen()

    def create_loading_page(self, text_l1=None, text_l2=None):
        """Create a loading page while GPT is running in the background"""
        # Create a new frame for the "loading" page
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        if text_l1 is None:
            text_l1 = "Great job! Just two more to go!"
        if text_l2 is None:
            text_l2 = "Please wait while we load the last two."

        # Page content
        loading_text_line1 = ctk.CTkLabel(master=self.frame, text=text_l1, font=("Open Sans", 40, "bold"))
        loading_text_line2 = ctk.CTkLabel(master=self.frame, text=text_l2, font=("Open Sans", 20))
        self.pattern = ctk.CTkLabel(master=self.frame, text='   ', font=("Consolas", 100))

        # Continue to display participant ID
        participantID = ctk.CTkLabel(master=self.frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=85, pady=10, side='bottom', anchor='w')

        loading_text_line1.pack(expand=False, anchor='center', pady=(100, 0))
        loading_text_line2.pack(expand=False, anchor='center')
        self.pattern.pack(expand=False, anchor='center', pady=(100, 0))

        # Define a loading text pattern and start updating the loading text
        self.loading_pattern = itertools.cycle([".  ", ".. ", "...", " ..", "  .", "   ", " . ", "   "])
        self.update_loading_text()

    def update_loading_text(self):
        """Update the loading text in the pattern given earlier"""
        next_text = next(self.loading_pattern)
        self.pattern.configure(text=next_text)
        self.frame.after(500, self.update_loading_text)

    def create_gpt_interaction_page(self):
        self.startTime = time.time()
        if self.gpt_refresh_code == -1:
            self.create_loading_page()
        else:
            self.create_loading_page("Loading","Please wait while we update your options.")

        # Test if there is a student profile already generated:
        if self.student_profile is None or self.gpt_refresh_code == 1:
            # Create profile generation thread
            profile_thread = threading.Thread(target=self.generate_profile)
            profile_thread.start()
        else:
            # Just continue on
            self.continue_gpt_interaction()

    def continue_gpt_interaction(self):
        print("Student Profile: " + self.student_profile)
        print("Opposite Student Profile: " + self.student_profile_opposite)
        try:
            if self.experiment_version == "Experimental":
                print("\nExperimental Group - using ACTUAL student profile to generate rewrites.")
            elif self.experiment_version == "Control":
                print("\nControl Group - using OPPOSITE student profile to generate rewrites.")
            else:
                raise ValueError("Invalid experiment version. Must be either 'Experimental' or Control'.")
        except Exception as e:
            print(f"Error: {e}")

        # Start a new thread to fetch GPT-generated text
        gpt_thread = threading.Thread(target=self.fetch_gpt_text)
        gpt_thread.start()

    def generate_profile(self):
        # Call on the code "openai_interact_profile". Use all student selections to create the profile.
        actual_profile, opposite_profile = openai_interact_profile.get_student_profile(self.user_selections,
                                                                                       self.user_notSelections,
                                                                                       self.paragraph_pair_list)

        self.student_profile = actual_profile
        self.student_profile_opposite = opposite_profile

        # Update GUI on main thread after generating the final profile.
        self.root.after(0, self.continue_gpt_interaction)

    def fetch_gpt_text(self):
        # Call on the code "openai_interact_rewrite" to get the responses from GPT to be displayed on the GPT screen.
        gpt_interaction_num = self.current_page - len(self.text) + self.num_gpt_interactions + 2
        try:
            if self.experiment_version == "Experimental" and self.gpt_refresh_code != 2:
                self.generated_text_iter1 = openai_interact_rewrite.get_gpt_response(self.student_profile,
                                                                                     gpt_interaction_num)
                gpt_interaction_num += 1
                self.generated_text_iter2 = openai_interact_rewrite.get_gpt_response(self.student_profile,
                                                                                     gpt_interaction_num)
            elif self.experiment_version == "Control" and self.gpt_refresh_code != 2:
                self.generated_text_iter1 = openai_interact_rewrite.get_gpt_response(self.student_profile_opposite,
                                                                                     gpt_interaction_num)
                gpt_interaction_num += 1
                self.generated_text_iter2 = openai_interact_rewrite.get_gpt_response(self.student_profile_opposite,
                                                                                     gpt_interaction_num)
            elif self.experiment_version == "Experimental" and self.gpt_refresh_code == 2:
                self.generated_text_iter2 = openai_interact_rewrite.get_gpt_response(self.student_profile,
                                                                                     gpt_interaction_num)

            elif self.experiment_version == "Control" and self.gpt_refresh_code == 2:
                self.generated_text_iter2 = openai_interact_rewrite.get_gpt_response(self.student_profile_opposite,
                                                                                     gpt_interaction_num)

            else:
                print(f"\nGUI failed on text generation.")
                raise ValueError("GUI failed on text generation. Invalid Experiment version.")
        except Exception as e:
            print(f"Error: {e}")

        self.gpt_paragraph1_iter1 = self.generated_text_iter1[0]
        self.gpt_paragraph2_iter1 = self.generated_text_iter1[1]

        self.gpt_paragraph1_iter2 = self.generated_text_iter2[0]
        self.gpt_paragraph2_iter2 = self.generated_text_iter2[1]

        # use after method to update the GUI on the main thread.
        if self.gpt_refresh_code == 2:
            self.root.after(0, self.update_gpt_interaction_page, self.gpt_paragraph1_iter2, self.gpt_paragraph2_iter2)
        else:
            self.root.after(0, self.update_gpt_interaction_page, self.gpt_paragraph1_iter1, self.gpt_paragraph2_iter1)

    def update_gpt_interaction_page(self, gpt_paragraph1, gpt_paragraph2):
        self.endTime = time.time()
        print(f"\n***\nTotal Loading Time: {self.endTime - self.startTime:.2f} s\n***\n")

        # Destroy the "loading" frame and update the page with GPT content
        if hasattr(self, 'frame'):
            self.frame.destroy()

        # Calculate the GPT interaction number
        gpt_interaction_num = self.current_page - len(self.text) + self.num_gpt_interactions + 2
        if gpt_interaction_num == 1:
            title = "Topic 5: Plate Tectonics"
        elif gpt_interaction_num == 2:
            title = "Topic 6: Electricity"
        else:
            title = "Title not supplied / Out of Range"

        # Set up radio button variable
        radio_var = tk.IntVar(value=0)

        # Save GPT paragraphs to self.text
        self.text[self.current_page] = [f"GPT Interaction {gpt_interaction_num}", f"{gpt_paragraph1}",
                                        f"{gpt_paragraph2}"]

        # Restart timer:
        self.startTime = time.time()

        # Call create_paragraph_page to build the layout
        self.create_paragraph_page(title, gpt_paragraph1, gpt_paragraph2, radio_var, gpt_interaction_num)

    def click_refresh(self, interaction_num):
        # Define common variables
        ttl = 'Confirm Refresh'
        msg = 'Refreshing generated response. OK?'

        if interaction_num != -1:
            print(f"Refresh called on GPT interaction number {interaction_num}.")
            confirmation = messagebox.askokcancel(title=ttl,
                                                  message=msg)
            if confirmation and interaction_num == 1:
                self.create_page(interaction_num)
            elif confirmation and interaction_num == 2:
                self.create_page(interaction_num)
            else:
                # If "cancel" is clicked, then return to the page.
                return
        elif interaction_num == -1:
            print("Refresh called on closing page.")
            msg += "\n\nCAUTION: Refreshing will erase any text entered into the textbox."
            confirmation = messagebox.askokcancel(title=ttl,
                                                  message=msg)
            if confirmation:
                # Destroy the previous frame if it exists
                if hasattr(self, 'frame'):
                    self.frame.destroy()
                self.create_close()
            else:
                # If "cancel" is clicked, then return to the page.
                return

    # Create a screen between the choosing paragraphs and the final screen where we will ask them to check profiler
    # outputs and provide feedback.
    def break_screen(self):
        # Destroy the previous paragraph pair screen if it exists.
        if hasattr(self, 'frame'):
            self.frame.destroy()

        # Display parameters:
        title = "Keep up the Great Work!"
        instruction_1 = ("On the next page, you will see two brief descriptions of your learning preferences based on "
                         "the choices you have made so far. You will be asked to select the paragraph which you "
                         "identify with most.")

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Create a header frame to hold and display instructions/title
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        # Pack in the instructions/title
        ctkTitle = ctk.CTkLabel(master=header_frame, text=title, font=('Open Sans', 30, 'bold'))
        ctkTitle.pack(padx=30, pady=(15, 20), anchor='n')

        ctkTitle2 = ctk.CTkLabel(master=header_frame,
                                 text=instruction_1,
                                 font=('Open Sans', 20), height=10, wraplength=650)

        # Add an image
        ctk_image = CTkImage(light_image=Image.open("Icons/welcome.png"), size=(500, 500))
        image_label = ctk.CTkLabel(master=header_frame, image=ctk_image, text='')
        image_label.pack(padx=0, pady=10)
        ctkTitle2.pack(padx=400, pady=(20, 0), anchor='s')

        # Button to submit and continue
        continue_button = ctk.CTkButton(master=self.frame, text="Continue",
                                        command=self.create_close,
                                        height=self.button_height,
                                        width=self.button_width,
                                        font=self.button_font)
        continue_button.pack(pady=(185, 40), anchor='s')

        # Display participant ID
        participantID = ctk.CTkLabel(master=self.frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=85, pady=10, side='bottom', anchor='w')

        # Start timer on break page.
        self.startTime = time.time()

    def create_close(self):
        # Destroy the "loading" frame and update the page with GPT content
        if hasattr(self, 'frame'):
            self.frame.destroy()

        # Display parameters
        title = "Which of the following two paragraphs best describes your learning preferences?"
        title2 = ("After making your selection, please briefly explain your reasoning. You "
                  "may type out your reasoning or, if you are more comfortable, you may explain your reasoning "
                  "verbally.")
        insert_instruct = ("Please explain your reasoning for your selection above in the textbox below. "
                           "You may either type your response or state your reasoning verbally.")

        # Page setup
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Create a page with two options for the student profile to be displayed. Should be randomized which is
        # presented left and right.
        self.clicked_text_box = None  # Reset tracking parameter to None each time a new page is created.

        # Create a header frame to hold and display instructions/title
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', padx=50, pady=(20, 0))

        # Pack in the instructions/title
        ctkTitle = ctk.CTkLabel(master=header_frame, text=title, font=('Open Sans', 24, 'bold'), height=10,
                                justify='left')
        ctkTitle.pack(padx=30, pady=(15, 5), anchor='nw')

        ctkTitle2 = ctk.CTkLabel(master=header_frame, text=title2, font=('Open Sans', 20), height=10, wraplength=1550,
                                 justify='left')
        ctkTitle2.pack(padx=30, pady=(0, 15), anchor='sw')

        # Set up a radio button variable and trace changes to it
        radio_var = tk.IntVar(value=0)

        # Create a divided frame for the profile options
        options_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        options_frame.pack(fill='both', expand=True, padx=60, pady=(10, 5))

        # Randomize order profiles will be displayed
        display_options = ['left', 'right']  # Defines the possibilities for position of actual profile
        display_choice = random.choice(display_options)

        if display_choice == 'left':
            left_profile_option = self.student_profile
            right_profile_option = self.student_profile_opposite
            order = "Actual profile on left, opposite profile on right."
        else:
            left_profile_option = self.student_profile_opposite
            right_profile_option = self.student_profile
            order = "Opposite profile on left, actual profile on right."
        print(display_choice + " --- " + order)

        # Create left option frame
        self.left_profile_frame = ctk.CTkFrame(options_frame, fg_color='transparent', corner_radius=15)
        self.left_profile_frame.pack(side='left', fill='both', expand=True, padx=10)

        # Left Profile label
        option_1_label = ctk.CTkLabel(master=self.left_profile_frame, text='Option 1', font=self.option_font)
        option_1_label.pack(padx=(0, 0), pady=(0, 0), anchor='n')

        # Left Profile option
        self.left_profile = ctk.CTkTextbox(self.left_profile_frame, font=self.text_font,
                                           corner_radius=self.text_box_radius, wrap="word", border_width=0,
                                           border_color="#DBDBDB", cursor="hand2")
        self.left_profile.insert(tk.END, left_profile_option)
        self.left_profile.configure(state='disabled')  # Disable textbox after all text is inserted
        self.left_profile.bind("<Button-1>", lambda event: radio_var.set(1))
        self.left_profile.pack(padx=(25, 25), pady=(5, 15), fill='both', expand=True)
        self.left_profile.bind("<Enter>", lambda event, t=self.left_profile: self.on_enter(event, t))
        self.left_profile.bind("<Leave>", lambda event, t=self.left_profile: self.on_leave(event, t))
        self.left_profile.bind("<Button-1>", lambda event,
                                                    t=self.left_profile: self.on_click_profile(event, t, radio_var))

        # Create right option frame
        self.right_profile_frame = ctk.CTkFrame(options_frame, fg_color='transparent', corner_radius=15)
        self.right_profile_frame.pack(side='right', fill='both', expand=True, padx=10)

        # Right Profile label
        option_2_label = ctk.CTkLabel(master=self.right_profile_frame, text='Option 2', font=self.option_font)
        option_2_label.pack(padx=(0, 0), pady=(0, 0), anchor='n')

        # Right Profile option
        self.right_profile = ctk.CTkTextbox(self.right_profile_frame, font=self.text_font,
                                            corner_radius=self.text_box_radius, wrap="word", border_width=0,
                                            border_color="#DBDBDB", cursor="hand2")
        self.right_profile.insert(tk.END, right_profile_option)
        self.right_profile.configure(state='disabled')  # Disable textbox after all text is inserted
        self.right_profile.bind("<Button-1>", lambda event: radio_var.set(2))
        self.right_profile.pack(padx=(25, 25), pady=(5, 15), fill='both', expand=True)
        self.right_profile.bind("<Enter>", lambda event, t=self.right_profile: self.on_enter(event, t))
        self.right_profile.bind("<Leave>", lambda event, t=self.right_profile: self.on_leave(event, t))
        self.right_profile.bind("<Button-1>", lambda event,
                                                     t=self.right_profile: self.on_click_profile(event, t, radio_var))

        # Create a new frame to display the response request.
        response_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        response_frame.pack(fill='both', expand=True, padx=10)

        response_title = ctk.CTkLabel(master=response_frame, text=insert_instruct, font=("Open Sans", 20))
        response_title.pack(anchor='w', padx=85, pady=(75,0))
        response = ctk.CTkTextbox(master=response_frame, height=140,
                                  font=self.text_font, corner_radius=10, wrap='word')
        response.pack(pady=(5, 5), padx=75, anchor='n', fill='x', expand=True)

        # Button to quit
        quit_button = ctk.CTkButton(self.frame, text="Submit",
                                    command=lambda: self.check_final_likert_response(radio_var.get(),
                                                                                     response.get("1.0",
                                                                                                  tk.END).strip()),
                                    height=self.button_height, width=self.button_width, font=self.button_font)
        quit_button.pack(pady=(10, 0))

        # Button to refresh closing page
        refresh_button_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        refresh_button_frame.pack(fill='both', side='bottom', padx=60, pady=0)

        # refresh_button = ctk.CTkButton(refresh_button_frame, text="Refresh", command=lambda: self.click_refresh(-1),
        #                               height=20, width=30, font=self.refresh_button_font,
        #                               fg_color=self.refresh_button_fg_color, hover_color=self.refresh_button_hover)
        # refresh_button.pack(padx=25, pady=10, side='right')

        participantID = ctk.CTkLabel(master=refresh_button_frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=25, pady=10, side='left')

    def on_click_profile(self, event, text_box, radio_var):
        # Unhighlight the previously clicked textbox if it exists
        if self.clicked_text_box and self.clicked_text_box != text_box:
            self.clicked_text_box.configure(border_color="#DBDBDB", border_width=0)

        # Update the clicked textbox and its appearance
        self.clicked_text_box = text_box
        text_box.configure(border_color="light blue", border_width=5)

        selected_value = radio_var.get()

        normal_color = "transparent"
        selected_color = "light blue"
        if selected_value == 1:
            # Highlight left frame and unhighlight right frame.
            self.left_profile_frame.configure(fg_color=selected_color)
            self.right_profile_frame.configure(fg_color=normal_color)
        elif selected_value == 2:
            # Highlight right frame and unhighlight left frame.
            self.left_profile_frame.configure(fg_color=normal_color)
            self.right_profile_frame.configure(fg_color=selected_color)

        # When the user clicks a radio button, save their choice and print it
        self.runtimes[self.current_page].append(time.time() - self.startTime)
        print(f"pg {self.current_page}   Option: {radio_var.get()}" +
              f"{self.runtimes[self.current_page][-1]:.2f} s")
        # Update "option_selected" to reflect that one of the two options has been selected.
        self.option_selected = True

    def click_next_page(self, response=None):
        '''Close the current page and launch the next page'''
        # Check if an option has been selected
        if not self.option_selected and self.current_page not in [0, len(self) - 1]:
            messagebox.showinfo(title="Selection Required", message="Please select an option before continuing.")
            return  # Do not move on to the next page and exit early.

        # Save endTime immediately
        self.endTime = time.time()
        self.calcTime = self.endTime - self.startTime

        # If an option has been selected, we need to append the selected text and not selected texts.
        # Save selected paragraphs to "self.user_selections" and the not selected paragraphs to
        # "self.user_notSelections".
        if 0 < self.current_page < len(self) - 1:
            selected_option = self.user_choices.get(self.current_page, None)
            notSelected_option = self.notUser_choices.get(self.current_page, None)
            self.user_selections.append(selected_option)
            self.user_notSelections.append(notSelected_option)
            self.user_selections_text.append(self.text[self.current_page][selected_option])
            self.user_notSelections_text.append(self.text[self.current_page][notSelected_option])

            print("\n***")
            print(f"Paragraphs Chosen List: {self.user_selections}")
            print(f"Paragraphs not Chosen List: {self.user_notSelections}")

            print(f"Paragraphs Chosen Text: {self.user_selections_text}")
            print(f"Paragraphs not Chosen Text: {self.user_notSelections_text}")
            print("***\n")

        # Save the runtime & response
        self.runtimes[self.current_page].append(self.calcTime)

        if response:
            self.user_choices[self.current_page] = response
            print(f"pg {self.current_page}   Final choice: Option {self.user_choices[self.current_page]}")
        if self.current_page == len(self) - 1:
            print(f"pg {self.current_page}   Quit              " +
                  f"{self.runtimes[self.current_page][-1]:.2f} s")
        if self.current_page == 0:
            print(f"Instr  Continue           " +
                  f"{self.runtimes[self.current_page][-1]:.2f} s\n")
        else:
            print(f"pg {self.current_page}   Continue           " +
                  f"{self.runtimes[self.current_page][-1]:.2f} s\nPage Submitted.")

        # Move forward
        self.current_page += 1  # Update the page number

        # Otherwise destroy the current frame and launch the next page
        if hasattr(self, 'frame'):
            self.frame.destroy()
        self.runtimes[self.current_page] = []  # Reset runtimes

        # Create the next page
        self.create_page()

    def check_final_likert_response(self, likert_rating, response_text):
        if likert_rating == 0:
            messagebox.showinfo(title="Selection Required",
                                message="You must select an option between the left and right profiles before saving "
                                        "and quitting.")
            return
        else:
            self.exit_page(likert_rating, response_text)

    def exit_page(self, likert_rating, response_text):
        if hasattr(self, 'frame'):
            self.frame.destroy()

        # Create a closing page similar to the welcome page
        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Create a header frame to hold and display instructions/title
        header_frame = ctk.CTkFrame(self.frame, fg_color='transparent')
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        # Pack in the instructions/title
        ctkTitle = ctk.CTkLabel(master=header_frame, text="Nice Job!", font=('Open Sans', 30, 'bold'))
        ctkTitle.pack(padx=30, pady=(15, 20), anchor='n')

        ctkTitle2 = ctk.CTkLabel(master=header_frame,
                                 text="You have finished this portion of the activity. We will now ask you to answer "
                                      "a few short questions about your experience using this tool.",
                                 font=('Open Sans', 20), height=10, wraplength=650)

        # Add an image
        ctk_image = CTkImage(light_image=Image.open("Icons/welcome.png"), size=(500, 500))
        image_label = ctk.CTkLabel(master=header_frame, image=ctk_image, text='')
        image_label.pack(padx=0, pady=10)
        ctkTitle2.pack(padx=30, pady=0, anchor='s')

        # Button to submit and continue
        continue_button = ctk.CTkButton(master=self.frame, text="Save & Exit",
                                        command=lambda: self.quit_program(likert_rating, response_text),
                                        height=self.button_height,
                                        width=self.button_width,
                                        font=self.button_font)
        continue_button.pack(pady=(185, 40), anchor='s')

        # Display participant ID
        participantID = ctk.CTkLabel(master=self.frame, text=self.participant_id_display,
                                     font=("Open Sans", 12))
        participantID.pack(padx=85, pady=10, side='bottom', anchor='w')


    def quit_program(self, likert_rating, response_text):
        if response_text == "":
            response_text = "Verbal response."
        print(f"\n\nQuit GUI called. Saving final outputs:\n*Profile Selection: {likert_rating}\n"
              f"Final Response Text: {response_text}\n\n*Recall: Profile selection = 1 is the left option, "
              f"= 2 is the right option.")
        self.quitGUI = time.time()
        print(f"\n\n---\nTotal GUI Runtime (From Instructions Screen): {(self.quitGUI - self.startGUI):.8} s")
        self.root.destroy()

    def set_text(self, page_num: int, title: str, text1: str, text2=""):
        '''Set the text for given page.
        Pages:
         page 0: the welcome page,
         pages 1-3: the initial three sets of paragraphs
         page 4: the GPT pages
         page 5: the close page.'''
        self.text[page_num] = [title, text1, text2]

    def __len__(self):
        '''Return the number of pages'''
        return len(self.text)


if __name__ == "__main__":
    import sys

    app = GUI()
    app.run()
