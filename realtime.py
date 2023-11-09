#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:27:23 2023

@author: joshualee
"""

import tkinter as tk
import customtkinter
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import os
import time
import subprocess
import sys

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'runs/train/exp7/weights/best.pt')

customtkinter.set_default_color_theme("dark-blue")

current_frame = None

start = False

detection_enabled = False



def toggle_detection():
    
    global detection_enabled
    
    detection_enabled = not detection_enabled
    
    btn_toggle_detection.configure(text="Start Detection" if not detection_enabled else "Stop Detection")
    status_message.set("Detection stopped." if not detection_enabled else "Detection running...")
    
    
    
def detect_and_display(frame):
    
    global status_message, start
    
    # Detect using your model
    results = model(frame)
    img = results.render()  # Image with detections
    img_array = np.array(img)
    img_array = img_array.squeeze(0)
    img = Image.fromarray(img_array)
    
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    
def save_image(folder_name,image):
    
    global status_message
    
    if image is None or start is False:
        print("No image to save")
        status_message.set("No image to save")
        root.update_idletasks()
        root.after(5000, lambda: status_message.set("Detection running...")) 
        return
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, f"img_{int(time.time())}.png")
    cv2.imwrite(file_path, image)
    print(f"Image saved to {file_path}")
    status_message.set(f"Image saved to {file_path}")
    root.update_idletasks()
    root.after(5000, lambda: status_message.set("Detection running..."))
    
def open_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)  # If the folder does not exist, create it
    try:
        if os.name == 'nt':  # For Windows
            os.startfile(path)
        elif sys.platform == 'darwin':  # For macOS
            subprocess.Popen(['open', path])
        else:
            print("OS not supported!")
    except Exception as e:
        print(e)
        status_message.set(f"Failed to open {path}")
        
def open_selected_folder():
    folder_name = folder_selected.get()
    if folder_name:
        open_folder(folder_name)
        
folders = ['correct_detections', 'false_positives', 'false_negatives']

def correct_detection():
    # Implement logic for correct detection
    save_image("correct_detections", current_frame)

def false_positive():
    # Save the image to "false_positives" directory
    save_image("false_positives", current_frame)

def false_negative():
    # Save the image to "false_negatives" directory
    save_image("false_negatives", current_frame)


# GUI setup

root = customtkinter.CTk()
#root = tk.Tk()
root.title("Ragwort Detector")
root.geometry(f"{1100}x{580}")
customtkinter.set_appearance_mode("Dark")

status_message = tk.StringVar()
folder_selected = tk.StringVar()

#frame = customtkinter.CTkFrame(master=root)
#frame = tk.Frame(root)
#frame.pack(padx=10, pady=10)

#root.grid_columnconfigure((1,2,3,4), weight=0)
#root.grid_rowconfigure(( 1, 2,3,4), weight=0)
root.grid_columnconfigure(0, weight=1)  # This will allow the left side to expand
root.grid_columnconfigure((1, 2, 3, 4), weight=1)  # These are your button columns
root.grid_columnconfigure(5, weight=1)  # This will allow the right side to expand

root.grid_rowconfigure(0, weight=1)  # This will allow the top side to expand
root.grid_rowconfigure((1, 2, 3), weight=0)  # The rows where your buttons are don't expand
root.grid_rowconfigure(4, weight=0)  # This row contains the status bar and doesn't expand

lmain = customtkinter.CTkLabel(master=root, text = '')
#lmain = tk.Label(frame)
#lmain.pack(pady=10)
lmain.grid(row=0,column=0, columnspan=6)

#btn_frame = customtkinter.CTkFrame(master=root)
#btn_frame = tk.Frame(root)
#btn_frame.pack(padx=10, pady=10)

btn_toggle_detection = customtkinter.CTkButton(master=root, text="Start Detection", command=toggle_detection)
#btn_toggle_detection = tk.Button(btn_frame, text="Start Detection", command=toggle_detection)
#btn_toggle_detection.pack(side="left", padx=10)
btn_toggle_detection.grid(row=3, column=1, sticky='s')

btn_correct = customtkinter.CTkButton(master = root, text="Correct", command=correct_detection)
#btn_correct = tk.Button(btn_frame, text="Correct", command=correct_detection)
#btn_correct.pack(side="left", padx=10)
btn_correct.grid(row=3, column=2, sticky='s')

btn_false_positive = customtkinter.CTkButton(master = root, text="False Positive", command=false_positive)
#btn_false_positive = tk.Button(btn_frame, text="False Positive", command=false_positive)
#btn_false_positive.pack(side="left", padx=10)
btn_false_positive.grid(row=3, column=3, sticky='s')

btn_false_negative = customtkinter.CTkButton(master = root, text="False Negative", command=false_negative)
#btn_false_negative = tk.Button(btn_frame, text="False Negative", command=false_negative)
#btn_false_negative.pack(side="left", padx=10)
btn_false_negative.grid(row=3, column=4, sticky='s')

statusbar = customtkinter.CTkLabel(master=root, textvariable=status_message)
#statusbar = tk.Label(root, textvariable=status_message, bd=1, relief=tk.SUNKEN, anchor=tk.W)
#statusbar.pack(side=tk.BOTTOM, fill=tk.X)
statusbar.grid(row=4, column=2, columnspan=2, sticky='s')

#folder_dropdown = tk.OptionMenu(root, folder_selected, *folders)
#folder_dropdown.grid(row=2, column=5, sticky='w')  # Adjust grid parameters as needed
optionmenu = customtkinter.CTkOptionMenu(master=root, variable=folder_selected, values=folders)
optionmenu.grid(row=2, column=5, sticky='w')

btn_open_folder = customtkinter.CTkButton(master=root, text="Open Folder", command=open_selected_folder)
btn_open_folder.grid(row=3, column=5, sticky='w')  # Adjust grid parameters as needed


cap = cv2.VideoCapture(0)  # 0 for default camera


def update():
    
    global start, current_frame, cap
    
    if detection_enabled:
    
        if start is False:
            
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                
            status_message.set("Detection running...")
            start = True
            
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                current_frame = frame.copy()
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detect_and_display(display_frame)
         
    else:
        
        if start:
            # Stop the camera when detection is toggled off
            if cap is not None:
                cap.release()
                cap = None
                lmain.configure(image='')  # Clear the display
            status_message.set("Detection stopped...")
            start = False
    
    root.after(10, update)
    
    
update()  # Start the update function
root.mainloop()

cap.release()