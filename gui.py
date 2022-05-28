from itertools import count
from sre_constants import SUCCESS
import tkinter as tk 
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
from cv2 import imshow
from sklearn.neighbors import NearestCentroid
from getkey import getkey, key
import pytesseract
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import csv
import time
import datetime
import sys
import shutil
from pushbullet import PushBullet
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile

access_token="o.fVGtMeajD1mIFzDei8cH8UPYWsYp0omo"

path = 'Images_Attendance'
images = []
classNames = []
deleted =[]
myList = os.listdir(path)
print("MyList")
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Classname",classNames)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    print("Enconding Done...........................")
    return encodeList

encodeListKnown=findEncodings(images)

def startpage(container):
    label = tk.Label(container, text ="Automatic Door Unlock System", font = "Helvetica", foreground="#263942")     #Add label
    label.config(font=("Helvetica", 15))    #set label size and font
    label.place(x = 100,y = 10) 
    
    def admin_clear_frame(frame):
        print(frame.winfo_children())   #get list of all childern widgets
        for widget in frame.winfo_children():    
            widget.destroy()        #removing widget
        
        admin(frame)    #calling admin function with empty frame as argument
    

    # opens the image 
    img = Image.open('static/door.png') 
    
    img = img.resize((180, 180), Image.ANTIALIAS) 
    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img) 
        # create a label 
    panel = tk.Label(container, image = img) 
        # set the image as img  
    panel.image = img 
    panel.place(x = 250 , y = 80)   #place the door image
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green')

    button1 = ttk.Button(container, text ="Admin",command = lambda : admin_clear_frame(container))  #call admin_clear_frame function on click
    button1.place(x = 95, y = 110)
    button2 = ttk.Button(container, text ="Doorbell",command = lambda : doorbell()) #call doorbell function to check authorised user or not
    button2.place(x = 95,y = 210)

def admin(container):
    
    label = tk.Label(container, text ="Admin Portal", font = "Helvetica", foreground="#263942") 
    label.config(font=("Helvetica", 15))
    label.place(x = 180,y = 20)
    
    img = Image.open('static/login.png')    
    img = img.resize((190, 190), Image.ANTIALIAS)   
    img = ImageTk.PhotoImage(img) 
    
    panel = tk.Label(container, image = img) 
    panel.image = img 
    panel.place(x = 230 , y = 80)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    

    def user_list_clear_frame(frame):
        for widget in frame.winfo_children():   #clearing frame
            widget.destroy()
        
        user_list(frame)    #call user_list with empty frame

    def new_user_clear_frame(frame):
        for widget in frame.winfo_children():   #clearing frame
            widget.destroy()    
        
        new_user(frame) #call new_user with empty frame

    def back_menu(frame):
        for widget in frame.winfo_children():   #clearing frame
            widget.destroy()

        startpage(frame)    #calling startpage

    button1 = ttk.Button(container, text ="Existing Users",command = lambda : user_list_clear_frame(container)) #call user_list_clear_frame on click
    button1.place(x = 82, y = 90)
    button2 = ttk.Button(container, text ="Add new User",command = lambda : new_user_clear_frame(container)) #call new_user_clear_frame on click
    button2.place(x = 82,y = 180)

    button3 = ttk.Button(container, text ="Back",command = lambda : back_menu(container))   #call back_menu on click
    button3.place(x = 82,y = 270)

def new_user(container):
    new_user = tk.StringVar()
    flag = tk.IntVar()
    flag.set(0)
    num_images = tk.IntVar()

    label = tk.Label(container, text ="New User Registeration", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 130,y = 20)
        
    name_label = tk.Label(container, text ="Name :", font = "Helvetica", foreground="#263942")
    name_label.config(font=("Helvetica", 12))
    name_label.place(x = 95,y = 90)

    def clear(frame):
        
        for widget in frame.winfo_children():
            widget.destroy()
        admin(frame)

    def check(container,name):
        if(name==""):
            return
        data = pd.read_csv('User.csv')
        if(name in list(data.Name)):
            messagebox.showerror("Error","User Name already Exists")    
            return

        if(name.upper() in deleted):
            deleted.remove(name.upper())
        print("------------After deletion---------------",deleted)
        data = pd.read_csv('User.csv')
        data.loc[len(data.Name)] = [name]
        data.set_index('Name',inplace=True)
        data.to_csv('User.csv')


        path2=os.getcwd() + f"/Images_Attendance/"
        path2=path2+str(name)
        vid = cv2.VideoCapture(0)
  
        while(True):
            
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
        
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(path2+".jpg",frame)
                break
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
              
        # encodeListKnown=findEncodings()
        messagebox.showinfo("Notification","Imaged Saved !")
        clear(container)
        
        curImg = cv2.imread(path2+".jpg")
        images.append(curImg)
        classNames.append(name)
        print("Classname",classNames)

        
        curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(curImg)[0]
        encodeListKnown.append(encode)
        print("Encodeing done after image upload")



    entry_name = tk.Entry(container,textvariable = new_user)    #take user name and save to new_user 
    print("Enter User Name:", new_user.get())
    entry_name.place(x = 165, y = 90)
    ttk.Style().configure("TButton", padding=6, relief="flat",background="#ccc",foreground='green') 
    
    
    button3 = ttk.Button(container, text ="Back",command = lambda : clear(container),state = tk.NORMAL) #adfter click go back to admin frame

    #Button to create dataset
    button1 = ttk.Button(container, text ="Upload Image",command = lambda : check(container,new_user.get())) 
    
    button1.place(x = 280, y = 180)
    button3.place(x = 70,y = 180)

def delete_selected(frame,Lb1): #delete selected user releted data
    a = Lb1.get(Lb1.curselection()).split(' ')
    print(a)    #['1.', 'Ajay']

    name=" ".join(a[1:])

    path2=os.getcwd() + f"/Images_Attendance/"
    path2=path2 + name +".jpg"
    pname=name.upper()
    deleted.append(pname)
    print("Printing Deleted List: ",deleted)
    os.remove(path2)    #s.remove() method in Python is used to remove or delete a file path
    
    data = pd.read_csv('User.csv')
    print("User Before: ",data)
    new_data = data[data.Name != name]  #Create new list which not contain selected user
    print("User After", new_data)
    new_data.set_index('Name',inplace = True) ## setting Name as index column
    print("New Dataset : ",new_data)
    new_data.to_csv('User.csv') #save new data to User.csv file
    
    for widget in frame.winfo_children():   #clear frame
        widget.destroy()        
        
    user_list(frame)    #show frame with updated list
    

def user_list(container):   #show user list and perform delete operation if require
    label = tk.Label(container, text ="List of Existing Users", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 140,y = 20)
    #names = []  
    Lb1 = tk.Listbox(container,selectbackground = "lightblue",yscrollcommand = True,bg = "#ccc")    #change background colur for selected user

    data = pd.read_csv('User.csv')  #get user list from user.csv
    names = list(data.Name)
    print("User present in User.csv file")
    print(names)    
    for i in range(len(names)):
        Lb1.insert(i+1, f"{i+1}. {names[i]}")        
    
    Lb1.place(x = 90,y = 80) 
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    def back_clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        admin(frame)

    button1 = ttk.Button(container, text ="Delete", 
							command = lambda : delete_selected(container,Lb1))  #call delete_seletected function with given conatainer 
    button1.place(x = 300, y = 120)

    button1 = ttk.Button(container, text ="Back", 
							command = lambda : back_clear_frame(container)) #call back_clear_frame function
    button1.place(x = 300, y = 180)

def sendNotification(personName,personImg):
    # Get the instance using access token
    pb = PushBullet(access_token)
    # Send the data by passing the main title
    # and text to be send
    time_now = datetime.datetime.now()
    push1 = pb.push_note("WHO IS THERE",personName+":"+str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second))
    
    with open(personImg, "rb") as pic:
        file_data = pb.upload_file(pic, personName+":"+str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second))
    #file_data = self.pb.upload_file(imagedata, 'Motion detected: ' + personImg)

    push = pb.push_file(**file_data)
    # push2 = pb.push_file("Person",personImg, file_type="image/jpeg")
    # Put a success message after sending
    # the notification
    print("Message sent successfully...")

def doorbell():
        flg=0
        userName="Unauthorised"
        userImg=None
        data = pd.read_csv("User.csv")
        names = list(data.Name)
        cap = cv2.VideoCapture(0)
        counter=0
        while True:
            success, img = cap.read()
            userImg=img
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print("FACE ID:",faceDis)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    r=0
                    b=0
                    g=255
                    if(name in deleted):
                        r=0
                        b=255
                        g=0
                        print("Eelement found in deleted listt-------------------------")   
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (r, g, b), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    userName=name
                    flg=1
                    cv2.waitKey(2000)
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                else:
                    name = "UNKNOWN"
                    print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  
                    if(counter>=9):
                        cv2.waitKey(2000)
                        cap.release()
                        cv2.destroyAllWindows()                  
                        flg=1
                        break
            counter=counter+1        
            
            if(flg==1 or counter>=10):
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            cv2.imshow('webcam', img)        
        time_now = datetime.datetime.now()
        path1 = os.getcwd() + f"/results/" #{name}{time_now}.jpg"
        #print(frame)
        #print(path)
        s = path1+str(userName) + str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second)
        cv2.imwrite(s+".jpg", userImg)
        # print("PATH: ",s)
        sendNotification(userName,s+".jpg")#    
        # cap.release()
        # cv2.destroyAllWindow()

app = tk.Tk()   #creating application main window
app.title("Who is there?")
app.geometry("450x350")     
app.resizable(False,False)
container = tk.Frame(app)   #It can be defined as a container to which, another widget can be added and organized
container.pack(side = "top", fill = "both", expand = True) #The Pack geometry manager packs widgets relative to the earlier widget.
container.grid_rowconfigure(0, weight = 1)
container.grid_columnconfigure(0, weight = 1)
startpage(container)    #call statrpage function  

def close():
    app.destroy()

def disable_event():
    pass
app.mainloop()
