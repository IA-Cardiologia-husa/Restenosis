import pandas as pd
import numpy as np
import sklearn
import pickle
from tkinter import *
from PIL import ImageTk, Image
class MyWindow:
    def __init__(self, win):
        self.lbl0=Label(win, text='Calculadora riesgo restenosis tras PCI', font='Helvetica 14 bold')
        self.lbl1=Label(win, text='Diabetes', font='Helvetica 10 bold')
        self.lbl2=Label(win, text='Enf. en >=2 vasos', font='Helvetica 10 bold')
        self.lbl3=Label(win, text='Trombo post_PCI', font='Helvetica 10 bold')
        self.lbl4=Label(win, text='TIMI post PCI', font='Helvetica 10 bold')
        self.lbl5=Label(win, text='Colesterol tot. anormal', font='Helvetica 10 bold')
        self.lbl6=Label(win, text='Plaquetas anormales', font='Helvetica 10 bold')
        self.resl=Label(win, text='Probabilidad restenosis', font='Helvetica 10 bold')

        self.num1 = IntVar()
        self.num2 = IntVar()
        self.num3 = IntVar()
        self.num4 = IntVar()
        self.num5 = IntVar()
        self.num6 = IntVar()

        self.num1.set(0)
        self.num2.set(0)
        self.num3.set(0)
        self.num4.set(3)
        self.num1.set(0)
        self.num1.set(0)

        rad1 = Radiobutton(window,text='Sí', variable=self.num1, value=1)
        rad2 = Radiobutton(window,text='No', variable=self.num1, value=0)
        rad1.place(x=250, y=200)
        rad2.place(x=300, y=200)
        rad3 = Radiobutton(window,text='Sí', variable=self.num2, value=1)
        rad4 = Radiobutton(window,text='No', variable=self.num2, value=0)
        rad3.place(x=250, y=250)
        rad4.place(x=300, y=250)
        rad5 = Radiobutton(window,text='Sí', variable=self.num3, value=1)
        rad6 = Radiobutton(window,text='No', variable=self.num3, value=0)
        rad5.place(x=250, y=300)
        rad6.place(x=300, y=300)
        rad7 = Radiobutton(window,text='1', variable=self.num4, value=1)
        rad8 = Radiobutton(window,text='2', variable=self.num4, value=2)
        rad81 = Radiobutton(window,text='3', variable=self.num4, value=3)
        rad7.place(x=250, y=350)
        rad8.place(x=300, y=350)
        rad81.place(x=350, y=350)
        rad9 = Radiobutton(window,text='Sí', variable=self.num5, value=1)
        rad10 = Radiobutton(window,text='No', variable=self.num5, value=0)
        rad9.place(x=250, y=400)
        rad10.place(x=300, y=400)
        rad11 = Radiobutton(window,text='Sí', variable=self.num6, value=1)
        rad12 = Radiobutton(window,text='No', variable=self.num6, value=0)
        rad11.place(x=250, y=450)
        rad12.place(x=300, y=450)
        self.resr=Entry()
        self.lbl0.place(x=55,y=5)
        self.lbl1.place(x=100, y=200)
        self.lbl2.place(x=100, y=250)
        self.lbl3.place(x=100, y=300)
        self.lbl4.place(x=100, y=350)
        self.lbl5.place(x=100, y=400)
        self.lbl6.place(x=100, y=450)
        self.b1=Button(win, text='Ejecutar modelo', command=self.add, font='Helvetica 14 bold')
        self.b1.place(x=100, y=500)
        self.resl.place(x=100, y=550)
        self.resr.place(x=300, y=550)
    def add(self):
        model = pickle.load(open('../models/finalized_model.sav', 'rb'))
        model = model.named_steps['clf']
        X = np.array([self.num1.get(), self.num2.get(), self.num3.get(), self.num4.get(), self.num5.get(), self.num6.get()]).reshape(1,6)
        proba = model.predict_proba(X)
        self.resr.delete(0, 'end')
        result=proba[0][1]
        self.resr.insert(END, str(result))
window=Tk()
mywin=MyWindow(window)
window.title('Calculadora riesgo restenosis tras PCI')
window.geometry("540x600+10+10")

load = Image.open("USAL.jpg")
render = ImageTk.PhotoImage(load)
img = Label(window, image=render)
img.image = render
img.place(x=10, y=40)


window.mainloop()
