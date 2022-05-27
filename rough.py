'''
def train_model(name):
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}


    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

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
    clf.write("./data/classifiers/"+name+"_classifier.xml")



def train_model(container,name,button3):
    img_train = face_recognition.load_image_file('train.png')
    img_train = cv2.cvtColor(img_train,cv2.COLOR_BGR2RGB)
    encodeTrain = face_recognition.face_encodings(img_train)[0]
    print(encodeTrain)
    button3['state'] = 'normal'
    np.save(f'dataset/{name}.npy',encodeTrain)
    messagebox.showinfo("Notification","Trained Dataset Succesfully")
    for widget in container.winfo_children():
            widget.destroy()
        
    admin(container)


def create_train(name):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except :
            pass
        key = 0
        if num_of_images > 310:
            break
    cv2.destroyAllWindows()
    print(num_of_images)
    return num_of_images


def create_dataset(button2,new_user,button3):
    global app
    print("Hey")
    print(button2)
    root = tk.Toplevel(app)
    root.geometry('330x330')
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    width, height = 300, 300
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    label = tk.Label(root)
    label.place(x = 0,y = 0)
    
    

    
        
    def capture(root,frame,new_user,button2,button3):
        print("Frame  is : ",frame)
        try:
            _, frame = cap.read()
            cv2.imwrite('train.png', frame)
            time.sleep(1)
            imgTest = face_recognition.load_image_file('train.png')
            imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
            faceLocTest = face_recognition.face_locations(imgTest)[0]
        
        except:
            messagebox.showerror("Error","Unable to recognize the face")
            root.destroy()
            return
        button2['state'] = 'normal'
        button3['state'] = 'disabled'
        np.save(f'dataset/{new_user.get()}.npy',np.array(2))
        data = pd.read_csv('User.csv')
        data.loc[len(data.index)] = [new_user.get()]
        data.set_index('Name',inplace = True)
        data.to_csv('User.csv')
        
        root.destroy()
    
    def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, show_frame)

    _, frame = cap.read()
    
    show_frame()
    button1 = ttk.Button(root, text ="Capture",command = lambda : capture(root,frame,new_user,button2,button3)) 
    button1.place(x = 120, y = 250)    
    root.mainloop()

def new_user(container):
    new_user = tk.StringVar()
    status = tk.IntVar()
    status.set(0)
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
        
    entry_name = tk.Entry(container,textvariable = new_user)
    entry_name.place(x = 165, y = 90)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    button3 = ttk.Button(container, text ="Back",command = lambda : clear(container),state = tk.NORMAL) 

    button2 = ttk.Button(container, text ="Train dataset",state = tk.DISABLED,command = lambda : train_model(container,new_user.get(),button3))
    

    #button1 = ttk.Button(container, text ="Create dataset",command = lambda : create_dataset(button2,new_user,button3)) 
    button1 = ttk.Button(container, text ="Create dataset",command = lambda : create_train(new_user.get())) 
    
    button1.place(x = 82, y = 180)

    button2.place(x = 210,y = 180)

    
    
    button3.place(x = 146, y = 260)







def doorbell(container):
    global app
    root = tk.Toplevel(app)
    root.geometry('330x330')
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    width, height = 300, 300
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    label = tk.Label(root)
    label.place(x = 0,y = 0)
    
    
    def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, show_frame)

    def test():
        try:
            _, frame = cap.read()
            cv2.imwrite('test.png', frame)
            time.sleep(1)
            imgTest = face_recognition.load_image_file('test.png')
            imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
            faceLocTest = face_recognition.face_locations(imgTest)[0]
            encodeTest = face_recognition.face_encodings(imgTest)[0]

        except:
            print("Error")
            return    

        path = os.getcwd() + "\dataset"
        faces = []
        for files in os.listdir(path):
            if os.path.isfile(os.path.join(path,files)):
                print(files)
                encode = np.load(f'dataset/{files}')
                faces.append(encode)

        print(faces)

        results = face_recognition.compare_faces(faces,encodeTest)
        print('results : ',results)
        person = "unknown"
        if(True in results):
            messagebox.showinfo("Notification",'User Detected Open door')
            person_index = results.index(True)
            data = pd.read_csv('User.csv')
            person = list(data.Name)[person_index]
            

        else:
            messagebox.showerror("Error",'User not recognized door lock')
        print(person)
        excel_data = pd.read_excel('entries.xlsx')
        excel_data.loc[len(excel_data)] = [person,datetime.datetime.now()]
        excel_data.to_excel('entries.xlsx',index = False)
        root.destroy()



    _, frame = cap.read()
    
    show_frame()
    
    root.after(2000,test)
    root.mainloop()

'''