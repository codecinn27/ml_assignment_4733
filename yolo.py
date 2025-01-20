#  pip install opencv-python Pillow torch pandas requests
#for torch must use pyton 10 or 11, else use virtual environment

import tkinter as tk
from tkinter import Tk 
from tkinter import filedialog
import cv2 #reading images and image processing
from PIL import Image, ImageTk #specific Image related library for TKinter
import numpy as np # processing in images after reading them into arrays
import os
from ultralytics import YOLO

root = Tk()
root.bind("<Escape>", lambda e: root.quit())
root.geometry("1400x1400")
root.title("Rock Paper Scissors Model")

# Load YOLOv5 model
MODEL_PATH = "models/yolov8_best.pt"  # Replace with your YOLOv8 weights path
model1 = YOLO(MODEL_PATH)
MODEL_PATH = "models/yolov11_best.pt"  # Replace with your YOLOv11 weights path
model2 = YOLO(MODEL_PATH)

def main_page():

    def confusion_matrix():
        def go_back_to_main_frame():
            display_frame1.place_forget()
            display_frame2.place_forget()
            back_frame.place_forget()
            main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)

        # Open and display the confusion matrix images
        main_frame.place_forget()
        cm1_image = Image.open("images/confusion_matrix/yolov8_confusion_matrix.png")
        cm2_image = Image.open("images/confusion_matrix/yolov11_confusion_matrix.png")

        # Resize the images to fit side by side
        cm1_image = cm1_image.resize((800, 680))
        cm2_image = cm2_image.resize((800, 680))

        cm1_imgtk = ImageTk.PhotoImage(image=cm1_image)
        cm2_imgtk = ImageTk.PhotoImage(image=cm2_image)

        display_frame1 = tk.Frame(root)
        display_frame1.place(relx=0.25, rely=0.5, width = 800, height = 680, anchor=tk.CENTER)

        display_frame1_label = tk.Label(display_frame1, text = "YOLO V8", font = ('Rockwell', 16), bg = "yellow")
        display_frame1_label.pack(side=tk.TOP)

        display_frame2 = tk.Frame(root)
        display_frame2.place(relx=0.75, rely=0.5, width = 800, height = 680, anchor=tk.CENTER)

        display_frame2_label = tk.Label(display_frame2, text = "YOLO V11", font = ('Rockwell', 16), bg = "yellow")
        display_frame2_label.pack(side=tk.TOP)

        # Display the images
        cm_label1 = tk.Label(display_frame1, image=cm1_imgtk)
        cm_label1.image = cm1_imgtk
        cm_label1.place(x=0, y=100)

        cm_label2 = tk.Label(display_frame2, image=cm2_imgtk)
        cm_label2.image = cm2_imgtk
        cm_label2.place(x=0, y=100)

        back_frame = tk.Frame(root)
        back_frame.place(relx=0, rely=0.05, width = 80, height = 30,)
        back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()

    def train_data():
        def go_back_to_main_frame():
             display_frame1.place_forget()
             display_frame2.place_forget()
             back_frame.place_forget()
             main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)

        # Open and display the confusion matrix images
        main_frame.place_forget()
        cm1_image = Image.open("images/train_result/yolov8.jpg")
        cm2_image = Image.open("images/train_result/yolov11.jpg")

        # Resize the images to fit side by side
        cm1_image = cm1_image.resize((1080, 240))
        cm2_image = cm2_image.resize((1080, 240))

        cm1_imgtk = ImageTk.PhotoImage(image=cm1_image)
        cm2_imgtk = ImageTk.PhotoImage(image=cm2_image)

        display_frame1 = tk.Frame(root)
        display_frame1.place(relx=0.5, rely=0.3, width = 1080, height = 240, anchor=tk.CENTER)

        display_frame1_label = tk.Label(display_frame1, text = "YOLO V8", font = ('Rockwell', 16), bg = "yellow")
        display_frame1_label.pack(side=tk.TOP)

        display_frame2 = tk.Frame(root)
        display_frame2.place(relx=0.5, rely=0.6, width = 1080, height = 240, anchor=tk.CENTER)

        display_frame2_label = tk.Label(display_frame2, text = "YOLO V11", font = ('Rockwell', 16), bg = "yellow")
        display_frame2_label.pack(side=tk.TOP)

        # Display the images
        cm_label1 = tk.Label(display_frame1, image=cm1_imgtk)
        cm_label1.image = cm1_imgtk
        cm_label1.place(x=0, y=100)

        cm_label2 = tk.Label(display_frame2, image=cm2_imgtk)
        cm_label2.image = cm2_imgtk
        cm_label2.place(x=0, y=100)

        back_frame = tk.Frame(root)
        back_frame.place(relx=0, rely=0.05, width = 80, height = 30,)
        back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()
    
    def web_cam_func():
        def go_back_to_main_frame():
             cap.release()
             for frame in [display_frame1, display_frame2, display_frame3, back_frame]:
                frame.place_forget()
             main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)
        
        main_frame.place_forget()
        width, height = 700, 700
        cap = cv2.VideoCapture(0) #access video file
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        display_frame1 = tk.Frame(root)
        display_frame1.place(relx=0.1, rely=0.1, width=500, height=400)

        display_frame1_label = tk.Label(display_frame1, text = "Original video", font = ('Rockwell', 16), bg = "yellow")
        display_frame1_label.pack(side=tk.TOP)

        display_frame2 = tk.Frame(root)
        display_frame2.place(relx=0.5, rely=0.1, width=500, height=400)

        display_frame2_label = tk.Label(display_frame2, text = "YOLO V8", font = ('Rockwell', 16), bg = "yellow")
        display_frame2_label.pack(side=tk.TOP)

        display_frame3 = tk.Frame(root)
        display_frame3.place(relx=0.1, rely=0.5, width=500, height=400)

        display_frame3_label = tk.Label(display_frame3, text="YOLO V11", font=('Rockwell', 16), bg="yellow")
        display_frame3_label.pack(side=tk.TOP)

        back_frame = tk.Frame(root)
        back_frame.place(relx=0, rely=0.05, width = 80, height = 30,)
        back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()


        lmain = tk.Label(display_frame1)
        lmain1 = tk.Label(display_frame2)
        lmain2 = tk.Label(display_frame3)
        lmain.place(x=0, y=50, width=500, height=400)
        lmain1.place(x=0, y=50, width=500, height=400)
        lmain2.place(x=0, y=50, width=500, height=400)
        
        def show_frame():
            _, frame = cap.read()
            frame2 = cv2.flip(frame, 1)
            frame3 =  frame2.copy()
            cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
                
            # Perform inference
            results1 = model1(frame2)
            for result in results1[0].boxes:
                xyxy = result.xyxy.cpu().numpy().astype(int)[0]
                conf = result.conf.cpu().item()
                cls = int(result.cls.cpu().item())
                label = f'{model1.names[cls]} {conf:.2f}'
                cv2.rectangle(frame2, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(frame2, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2image1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            img1 = Image.fromarray(cv2image1)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            lmain1.imgtk = imgtk1
            lmain1.configure(image=imgtk1)

            results2 = model1(frame2)
            for result in results2[0].boxes:
                xyxy = result.xyxy.cpu().numpy().astype(int)[0]
                conf = result.conf.cpu().item()
                cls = int(result.cls.cpu().item())
                label = f'{model2.names[cls]} {conf:.2f}'
                cv2.rectangle(frame3, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(frame3, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            img2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            lmain2.imgtk = imgtk2
            lmain2.configure(image=imgtk2)
                
            lmain.after(10, show_frame)
            
        show_frame()

    def upload_vid_func():
        def go_back_to_main_frame():
                browse_frame.place_forget()
                back_frame.place_forget()
                main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)

        def browse_file():
            def run_yolov5_on_video():
                def go_back_to_main_frame():
                    cap.release()
                    display_frame1.place_forget()
                    display_frame2.place_forget()
                    display_frame3.place_forget()
                    back_frame.place_forget()
                    main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)
                 
                browse_frame.place_forget()
                width, height = 700, 700
                #  print(file_path)
                cap = cv2.VideoCapture(file_path)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                display_frame1 = tk.Frame(root)
                display_frame1.place(relx=0.1, rely=0.1, width = 500, height = 400)

                display_frame1_label = tk.Label(display_frame1, text = "Original video", font = ('Rockwell', 16), bg = "yellow")
                display_frame1_label.pack(side=tk.TOP)

                display_frame2 = tk.Frame(root)
                display_frame2.place(relx=0.5, rely=0.1, width = 500, height = 400)

                display_frame2_label = tk.Label(display_frame2, text = "YOLOv8", font = ('Rockwell', 16), bg = "yellow")
                display_frame2_label.pack(side=tk.TOP)

                display_frame3 = tk.Frame(root)
                display_frame3.place(relx=0.1, rely=0.5, width = 500, height = 400)

                display_frame3_label = tk.Label(display_frame3, text = "YOLOv11", font = ('Rockwell', 16), bg = "yellow")
                display_frame3_label.pack(side=tk.TOP)

                back_frame = tk.Frame(root)
                back_frame.place(relx=0, rely=0.05, width = 80, height = 30,)
                back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
                back_button.pack()


                lmain = tk.Label(display_frame1)
                lmain1 = tk.Label(display_frame2)
                lmain2 = tk.Label(display_frame3)
                lmain.place(x=0, y=50, width=500, height=400)
                lmain1.place(x=0, y=50, width=500, height=400)
                lmain2.place(x=0, y=50, width=500, height=400)
        
                def show_frame():
                    
                    _, frame = cap.read()
                    frame2 = frame
                    frame3 =  frame2.copy()
                    cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)

                    imgtk = ImageTk.PhotoImage(image=img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    
                    # Perform inference
                    results1 = model1(frame2)
                    
                    for result in results1[0].boxes:
                        xyxy = result.xyxy.cpu().numpy().astype(int)[0]
                        conf = result.conf.cpu().item()
                        cls = int(result.cls.cpu().item())
                        label = f'{model1.names[cls]} {conf:.2f}'
                        cv2.rectangle(frame2, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                        cv2.putText(frame2, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    cv2image1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    img1 = Image.fromarray(cv2image1)

                    imgtk1 = ImageTk.PhotoImage(image=img1)
                    lmain1.imgtk = imgtk1
                    lmain1.configure(image=imgtk1)

                    # Perform inference with Model 2
                    results2 = model2(frame2)
      
                    for result in results2[0].boxes:
                        xyxy = result.xyxy.cpu().numpy().astype(int)[0]
                        conf = result.conf.cpu().item()
                        cls = int(result.cls.cpu().item())
                        label = f'{model2.names[cls]} {conf:.2f}'
                        cv2.rectangle(frame3, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                        cv2.putText(frame3, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)
                    imgtk2 = ImageTk.PhotoImage(image=img2)
                    lmain2.imgtk = imgtk2
                    lmain2.configure(image=imgtk2)
                
                    lmain.after(1, show_frame)
                
                show_frame()

            filename = filedialog.askopenfilename(filetypes=[("video files", "*.*")])
            file_path = os.path.abspath(filename)

            run_yolov5_on_video()

        back_frame = tk.Frame(root)
        back_frame.place(relx=0, rely=0.05, width = 80, height = 30,)
        back_button = tk.Button(back_frame, text= "BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()
        
        main_frame.place_forget()

        browse_frame = tk.Frame(root, bg = "orange")
        browse_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        browse_button = tk.Button(browse_frame, text="Browse", font= ("Rockwell", 20), bg="Yellow", fg="white", command=browse_file)
        browse_button.pack()

    main_frame = tk.Frame(root)

    main_frame.place(relx=0.5, rely=0.5, width = 500, height = 400, anchor=tk.CENTER)
    
    web_cam = tk.Button(main_frame, text = "Web cam", command = web_cam_func, bg = "yellow", fg = "purple", font=('Rockwell', 18))
    
    web_cam.place(x = 10, y = 100, width = 200)
    
    upload_vid = tk.Button(main_frame, text = "Upload Video", command = upload_vid_func, bg = "yellow", fg = "purple", font=('Rockwell', 18))
    
    upload_vid.place(x = 300, y = 100, width = 200)

    # New button for Confusion Matrix
    confusion_matrix_btn = tk.Button(main_frame, text="Confusion Matrix", command=confusion_matrix, bg="yellow", fg="purple", font=('Rockwell', 18))
    confusion_matrix_btn.place(x=10, y=200, width = 200)

    train_result_btn = tk.Button(main_frame, text="Train Result", command=train_data, bg="yellow", fg="purple", font=('Rockwell', 18))
    train_result_btn.place(x=300, y=200, width = 200)

main_page()

Title_label = tk.Label(root, text = "YOLOv8 vs YOLOv11 Object detection", font = ('Rockwell', 20), bg = "yellow")
Title_label.pack(side=tk.TOP)

# Execute tkinter
root.mainloop()