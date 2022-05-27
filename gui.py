import tkinter as tk 
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
import pytesseract
import cv2
import numpy as np
#import face_recognition
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

access_token="o.fVGtMeajD1mIFzDei8cH8UPYWsYp0omo"

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

    def check(container,name,flag,button1,button2,button3,num_images):
        data = pd.read_csv('User.csv')

        if(name in list(data.Name)):
            messagebox.showerror("Error","User Name already Exists")    
            return
        create_dataset(container,name,flag,button1,button2,button3,num_images)  #create dataset for user
        return

    def build_model(name,button1,button2,button3):
        entry_name.delete(0,'end')   #

        train_model(name,button1,button2,button3)   


    entry_name = tk.Entry(container,textvariable = new_user)    #take user name and save to new_user 
    print("Enter User Name:", new_user.get())
    entry_name.place(x = 165, y = 90)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    
    button3 = ttk.Button(container, text ="Back",command = lambda : clear(container),state = tk.NORMAL) #adfter click go back to admin frame

    #Button to train dataset. Initially disable...enable after the creating dataset
    button2 = ttk.Button(container, text ="Train dataset",state = tk.DISABLED,command = lambda : build_model(new_user.get(),button1,button2,button3))
  
    #Button to create dataset
    button1 = ttk.Button(container, text ="Create dataset",command = lambda : check(container,new_user.get(),flag,button1,button2,button3,num_images)) 
    
    button1.place(x = 310, y = 180)
    button2.place(x = 180,y = 180)
    button3.place(x = 50,y = 180)



def create_dataset(container,name,flag,button1,button2,button3,num_images):
    path = "./dataset/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")  #loading require haar-cascade XML file
    try:
        os.makedirs(path)   #create folder for user
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)   #define video capture object
    while True:
        ret, img = vid.read()   #capture the video frame 
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #Initially image is a three-layer image i.e RGB so it converted to one layer image i.e. grayimage
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.2, minNeighbors=5)   #this method return boundary rectangles for detected face  
        key = 0
        for x, y, w, h in face: #iterating through reactangle of detected faces
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))
            new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)      #display image
            key = cv2.waitKey(1) & 0xFF
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)  #store image in dateset/user folder
            num_of_images += 1
        except :
            pass
        
        if num_of_images > 300:
            break
    cv2.destroyAllWindows() #destroy all window
    print(num_of_images)
    button2['state'] = "normal"
    button3['state'] = 'disabled'
    button1['state'] = 'disabled'
    flag.set(1)
    num_images.set(num_of_images)
    print(flag.get())
    app.protocol("WM_DELETE_WINDOW",disable_event)     #disabling the x button to close the window
    s = f"Images captuared : {num_images.get()}"
    label1 = tk.Label(container, text = s, font = "Helvetica", foreground="red")
    label1.config(font=("Helvetica", 12))
    label1.place(x = 150,y = 250)
    return


def train_model(name,button1,button2,button3):
    path = os.path.join(os.getcwd()+"/dataset/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}

    for root,dirs,files in os.walk(path):
        pictures = files


    for pic in pictures :
        imgpath = path+pic
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        #names[name].append(id)
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
    #Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./classifiers/"+name+"_classifier.xml")
    button2['state'] = 'disabled'
    button3['state'] = "normal"
    button1['state'] = "normal"
    app.protocol("WM_DELETE_WINDOW",close)

    data = pd.read_csv('User.csv')
    data.loc[len(data.Name)] = [name]
    data.set_index('Name',inplace=True)
    
    data.to_csv('User.csv')
    
    messagebox.showinfo("Notififcation","Succesfully Trained the model")


def delete_selected(frame,Lb1): #delete selected user releted data
    a = Lb1.get(Lb1.curselection()).split(' ')
    print(a)    #['1.', 'Ajay']
    path = os.getcwd()
    
    path1 = path + f"/dataset/{a[1]}"
    path2 = path + f'/classifiers/{a[1]}_classifier.xml' 
    #print(path,path1,path2)
    shutil.rmtree(path1) #shutil.rmtree() is used to delete an entire directory tree, path must point to a directory 
    os.remove(path2)    #s.remove() method in Python is used to remove or delete a file path
    
    data = pd.read_csv('User.csv')
    print("User Before: ",data)
    new_data = data[data.Name != a[1]]  #Create new list which not contain selected user
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
        data = pd.read_csv("User.csv")
        names = list(data.Name)
        face_cascade = cv2.CascadeClassifier('./static/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        cap = cv2.VideoCapture(0)
        userImg=None
        for i in names:
            print(i)
            name = i
            recognizer.read(f"./classifiers/{name}_classifier.xml")
            pred = 0
            for i in range(50):
                
                ret, frame = cap.read()
                #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                unknownPerson=frame
                for (x,y,w,h) in faces:


                    roi_gray = gray[y:y+h,x:x+w]

                    id,confidence = recognizer.predict(roi_gray)
                    confidence = 100 - int(confidence)
                    
                    if confidence > 60:
                        #if u want to print confidence level
                                #confidence = 100 - int(confidence)
                                pred = pred+1
                                text = name.upper()
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                                print("Matched Face")
                                
                                if(pred == 10):
                                    time_now = datetime.datetime.now()
                                    path = os.getcwd() + f"/results/" #{name}{time_now}.jpg"
                                    #print(frame)
                                    #print(path)
                                    s = path+str(name) + str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second)
                                    cv2.imwrite(s+".jpg", frame)
                                    cv2.waitKey(2000)
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    excel_data = pd.read_excel('entries.xlsx')
                                    excel_data.loc[len(excel_data)] = [name,datetime.datetime.now()]
                                    excel_data.to_excel('entries.xlsx',index = False)
                                    # userImg=Image.open(s+".jpg")
                                    sendNotification(name,s+".jpg")#send notification
                                    messagebox.showinfo("Notification","User Detected Open the door")
                                    return    
                    else:   
                                #pred += -1
                                text = "UnknownFace"
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

                cv2.imshow("image", frame)


                if cv2.waitKey(20) & 0xFF == ord('q'):
                    print(pred)
                    
        messagebox.showerror("Error","Unauthorized Person doors are closed")
        time_now = datetime.datetime.now()
        path = os.getcwd() + f"/results/" #{name}{time_now}.jpg"
        #print(frame)
        #print(path)
        s = path+str(name) + str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second)
        cv2.imwrite(s+".jpg", frame)
        sendNotification("Unknown_Person",s+".jpg")#send notification

        cap.release()
        cv2.destroyAllWindows()




app = tk.Tk()   #creating application main window
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