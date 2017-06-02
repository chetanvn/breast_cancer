from tkinter import *
from main import predictor
root = Tk()


#Radio button
radbutton = StringVar()
radiobutton1=Radiobutton(root, text="Naive Bayes", variable=radbutton, value="gnb")
radiobutton2=Radiobutton(root, text="KNN", variable=radbutton, value="rand_knn")
radiobutton3=Radiobutton(root, text="Decision Tree", variable=radbutton, value="rand_dt")
radiobutton4=Radiobutton(root, text="Support Vector Machine", variable=radbutton, value="grid_sv")
radiobutton5=Radiobutton(root, text="Random Forest", variable=radbutton, value="rand_rf")
radiobutton6=Radiobutton(root, text="Neural Network", variable=radbutton, value="grid_nn")


radiobutton1.pack()
radiobutton2.pack()
radiobutton3.pack()
radiobutton4.pack()
radiobutton5.pack()
radiobutton6.pack()

#Labels


#Entries
inputvar1 = StringVar()
inputvar2 = StringVar()
inputvar3 = StringVar()
inputvar4 = StringVar()
inputvar5 = StringVar()
inputvar6 = StringVar()
inputvar7 = StringVar()
inputvar8 = StringVar()
inputvar9 = StringVar()

entry1 = Entry(root, textvariable=inputvar1)
entry2 = Entry(root, textvariable=inputvar2)
entry3 = Entry(root, textvariable=inputvar3)
entry4 = Entry(root, textvariable=inputvar4)
entry5 = Entry(root, textvariable=inputvar5)
entry6 = Entry(root, textvariable=inputvar6)
entry7 = Entry(root, textvariable=inputvar7)
entry8 = Entry(root, textvariable=inputvar8)
entry9 = Entry(root, textvariable=inputvar9)

entry1.pack()
entry2.pack()
entry3.pack()
entry4.pack()
entry5.pack()
entry6.pack()
entry7.pack()
entry8.pack()
entry9.pack()


#Function activated by button
def sayhi():
    print (radbutton.get())
    predictor(radbutton.get(),float(entry1.get()),float(entry2.get()),float(entry3.get()),float(entry4.get()),float(entry5.get()),float(entry6.get()),float(entry7.get()),float(entry8.get()),float(entry9.get()))


#Button
button1 = Button(root, command=sayhi, text="Predict")
button1.pack()


root.mainloop()