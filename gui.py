from tkinter import *
from main import predictor
from tkinter import messagebox
from PIL import ImageTk, Image
root = Tk()

#image
img=ImageTk.PhotoImage(Image.open("cancer.png"))
panel1 = Label(root, image = img)
panel1.pack(side = "top", fill = "both", expand = "no")
#Radio button
radbutton = StringVar()
radiobutton1=Radiobutton(root, text="Naive Bayes", variable=radbutton, value="gnb", width = 20, height=1)
radiobutton2=Radiobutton(root, text="KNN", variable=radbutton, value="rand_knn", height=1)
radiobutton3=Radiobutton(root, text="Decision Tree", variable=radbutton, value="rand_dt", height=1)
radiobutton4=Radiobutton(root, text="Support Vector Machine", variable=radbutton, value="grid_sv", height=1)
radiobutton5=Radiobutton(root, text="Random Forest", variable=radbutton, value="rand_rf", height=1)
radiobutton6=Radiobutton(root, text="Neural Network", variable=radbutton, value="grid_nn", height=1)


radiobutton1.pack()
radiobutton2.pack()
radiobutton3.pack()
radiobutton4.pack()
radiobutton5.pack()
radiobutton6.pack()


#Entries and labels
inputvar1 = StringVar()
inputvar2 = StringVar()
inputvar3 = StringVar()
inputvar4 = StringVar()
inputvar5 = StringVar()
inputvar6 = StringVar()
inputvar7 = StringVar()
inputvar8 = StringVar()
inputvar9 = StringVar()

label0 = Message(root, text="===========================",width=220)
label1 = Message(root, text="Radius Mean",width=100,pady=10)
label2 = Message(root, text="Texture Mean",width=100)
label3 = Message(root, text="Perimeter Mean",width=100)
label4 = Message(root, text="Area Mean",width=100)
label5 = Message(root, text="Compactness Mean",width=110)
label6 = Message(root, text="Concavity Mean",width=100)
label7 = Message(root, text="Concave Points Mean",width=115)
label8 = Message(root, text="Area Worst",width=100)
label9 = Message(root, text="Perimeter Worst",width=100)

entry1 = Entry(root, textvariable=inputvar1)
entry2 = Entry(root, textvariable=inputvar2)
entry3 = Entry(root, textvariable=inputvar3)
entry4 = Entry(root, textvariable=inputvar4)
entry5 = Entry(root, textvariable=inputvar5)
entry6 = Entry(root, textvariable=inputvar6)
entry7 = Entry(root, textvariable=inputvar7)
entry8 = Entry(root, textvariable=inputvar8)
entry9 = Entry(root, textvariable=inputvar9)

label0.pack()
label1.pack()
entry1.pack()
label2.pack()
entry2.pack()
label3.pack()
entry3.pack()
label4.pack()
entry4.pack()
label5.pack()
entry5.pack()
label6.pack()
entry6.pack()
label7.pack()
entry7.pack()
label8.pack()
entry8.pack()
label9.pack()
entry9.pack()

#T= Text(root, height=2,width=10)

#Function activated by button
def sayhi():
    print (radbutton.get())
    output1=predictor(radbutton.get(),float(entry1.get()),float(entry2.get()),float(entry3.get()),float(entry4.get()),float(entry5.get()),float(entry6.get()),float(entry7.get()),float(entry8.get()),float(entry9.get()))
    global T
    if output1==[1]:
        #T.insert(END, "Malignant")
        messagebox.showinfo("Result","Malignant")
    else:
        #T.insert(END, "Benign")
        messagebox.showinfo("Result", "Benign")

#Button

labelp = Message(text="===========================",width=220)
labelp2 = Message(text="===========================",width=220)
button1 = Button(root, command=sayhi, text="Predict")
labelp.pack()
button1.pack()
labelp2.pack()
#T.pack()

root.mainloop()