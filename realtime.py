#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:27:23 2023

@author: joshualee
"""
#Import modules
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

def detection_switch():
    
    global detection_enabled
    
    detection_enabled = not detection_enabled
    
    btn_detection.configure(text="Start Detection" if not detection_enabled else "Stop Detection")
    status_message.set("Detection stopped." if not detection_enabled else "Detection running...")
    
    
    
def display(frame):
    
    global status_message, start
    
    # detect using the model
    results = model(frame)
    img = results.render()  # bounding boxes
    img_array = np.array(img)
    img_array = img_array.squeeze(0)
    img = Image.fromarray(img_array)
    
    imgtk = ImageTk.PhotoImage(image=img)
    camera_display.imgtk = imgtk
    camera_display.configure(image=imgtk)
    
    
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
        os.makedirs(path)  # create folder if does not exist
        
    if sys.platform == 'darwin':  # macOS specific
        subprocess.Popen(['open', path])
    
    else:
        print("Operating system not supported")
        
        
def open_selected_folder():
    folder_name = folder_selected.get()
    if folder_name:
        open_folder(folder_name)
        
folders = ['Correct Detections', 'False Positives', 'False Negatives']

def correct_detection():
   
    save_image("Correct Detections", current_frame)

def false_positive():
    
    save_image("False Positives", current_frame)

def false_negative():
    
    save_image("False Negatives", current_frame)


# GUI setup

root = customtkinter.CTk()
root.title("Ragwort Detector")
root.geometry(f"{1100}x{580}")
customtkinter.set_appearance_mode("Dark")

status_message = tk.StringVar()
folder_selected = tk.StringVar()

root.grid_columnconfigure(0, weight=1) # weight of 1 means it can expand, 0 means it cannot
root.grid_columnconfigure((1, 2, 3, 4), weight=1)
root.grid_columnconfigure(5, weight=1)

root.grid_rowconfigure(0, weight=1)  
root.grid_rowconfigure((1, 2, 3), weight=0)
root.grid_rowconfigure(4, weight=0)

camera_display = customtkinter.CTkLabel(master=root, text = '')
camera_display.grid(row=0,column=0, columnspan=6)

btn_detection = customtkinter.CTkButton(master=root, text="Start Detection", command=detection_switch)
btn_detection.grid(row=3, column=1, sticky='s')

btn_correct = customtkinter.CTkButton(master = root, text="Correct", command=correct_detection)
btn_correct.grid(row=3, column=2, sticky='s')

btn_false_positive = customtkinter.CTkButton(master = root, text="False Positive", command=false_positive)
btn_false_positive.grid(row=3, column=3, sticky='s')

btn_false_negative = customtkinter.CTkButton(master = root, text="False Negative", command=false_negative)
btn_false_negative.grid(row=3, column=4, sticky='s')

statusbar = customtkinter.CTkLabel(master=root, textvariable=status_message)
statusbar.grid(row=4, column=2, columnspan=2, sticky='s')

optionmenu = customtkinter.CTkOptionMenu(master=root, variable=folder_selected, values=folders)
optionmenu.grid(row=2, column=5, sticky='w')

btn_open_folder = customtkinter.CTkButton(master=root, text="Open Folder", command=open_selected_folder)
btn_open_folder.grid(row=3, column=5, sticky='w')


cap = cv2.VideoCapture(0)  # 0 for webcam


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
                display(display_frame)
         
    else:
        
        if start:
            # stop camera when detection is off
            if cap is not None:
                cap.release()
                cap = None
                camera_display.configure(image='')  # clear camera display
            status_message.set("Detection stopped...")
            start = False
    
    root.after(10, update)
    
    
update() # starts function
root.mainloop()

cap.release()