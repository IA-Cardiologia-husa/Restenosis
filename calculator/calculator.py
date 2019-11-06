import pandas as pd
import numpy as np
import sklearn
import pickle
from tkinter import *
from PIL import ImageTk, Image
from funcs_bokeh import *
from bokeh.io import export_png

class MyWindow:
    def __init__(self, win):
        self.lbl0=Label(win, text='Calculadora riesgo restenosis tras PCI', font='Helvetica 14 bold')
        self.lbl1=Label(win, text='Diabetes', font='Helvetica 10 bold')
        self.lbl2=Label(win, text='Enf. en >=2 vasos', font='Helvetica 10 bold')
        self.lbl3=Label(win, text='Trombo post PCI', font='Helvetica 10 bold')
        self.lbl4=Label(win, text='TIMI post PCI', font='Helvetica 10 bold')
        self.lbl5=Label(win, text='Colesterol tot. anormal', font='Helvetica 10 bold')
        self.lbl6=Label(win, text='Plaquetas anormales', font='Helvetica 10 bold')
        self.resl=Label(win, text='Score restenosis', font='Helvetica 10 bold')
        self.SPE=Label(win, text='Especificidad', font='Helvetica 10 bold')
        self.REC=Label(win, text='Sensibilidad', font='Helvetica 10 bold')
        self.PRE=Label(win, text='Exhaustividad', font='Helvetica 10 bold')
        self.NPV=Label(win, text='Valor predictivo negativo', font='Helvetica 10 bold')

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


        rad50 = Radiobutton(window,text='0', variable=self.num3, value=0)
        rad51 = Radiobutton(window,text='1', variable=self.num3, value=1)
        rad52 = Radiobutton(window,text='2', variable=self.num3, value=2)
        rad53 = Radiobutton(window,text='3', variable=self.num3, value=3)
        rad54 = Radiobutton(window,text='4', variable=self.num3, value=4)
        rad55 = Radiobutton(window,text='5', variable=self.num3, value=5)
        rad50.place(x=250, y=300)
        rad51.place(x=280, y=300)
        rad52.place(x=310, y=300)
        rad53.place(x=340, y=300)
        rad54.place(x=370, y=300)
        rad55.place(x=400, y=300)


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
        data = pd.read_csv('../models/probs.csv', sep = ';')
        PREC, REC, SPE, NPV = point_scores(data, proba[0][1])
        self.resr.delete(0, 'end')
        result=proba[0][1]
        fpr1, tpr1, prec1, rec1, spe1, npv1 = gracia_curves(data)
        thresholds_pr = [0, 0.18, 0.56, 0.81,1]
        thresholds_fpr = [0, 0.018, 0.14, 0.466,1]
        thresholds_npv = [0, 0.53, 0.86, 0.98,1]
        colors = ["red", "orange", "yellow", "green"]
        fig1 = figure_bokeh('ROC', thresholds=thresholds_fpr,
                            x_label = '1 - Especificidad', y_label = 'Sensitividad',
                            x = fpr1, y = tpr1, x_p = 1-SPE, y_p = REC, colors = colors, legend_pos = 'bottom_right')
        fig2 = figure_bokeh('P-R', thresholds=thresholds_pr, x_label = 'Sensitividad',
                            y_label = 'Exhaustividad', x = rec1, y = prec1,
                            x_p = REC, y_p = PREC, colors = colors,
                            legend_pos = 'top_right')
        fig3 = figure_bokeh('NPV-SPE', thresholds=thresholds_npv,
                            x_label = 'Especificidad',
                            y_label = 'Valor Predictivo Negativo',
                            x = spe1, y = npv1, x_p = SPE, y_p = NPV,
                            colors = colors[::-1], legend_pos = 'bottom_right',
                            range_y = (0.85,1))
        export_png(fig1, filename="fig1.png")
        export_png(fig2, filename="fig2.png")
        export_png(fig3, filename="fig3.png")

        load2 = Image.open("fig1.png")
        render2 = ImageTk.PhotoImage(load2)
        img2 = Label(window, image=render2)
        img2.image = render2
        img2.place(x=550, y=40)

        load3 = Image.open("fig2.png")
        render3 = ImageTk.PhotoImage(load3)
        img3 = Label(window, image=render3)
        img3.image = render3
        img3.place(x=550, y=350)

        load4 = Image.open("fig3.png")
        render4 = ImageTk.PhotoImage(load4)
        img4 = Label(window, image=render4)
        img4.image = render4
        img4.place(x=900, y=350)

        self.SPE.place(x=920, y=100)
        self.REC.place(x=920, y=150)
        self.PRE.place(x=920, y=200)
        self.NPV.place(x=920, y=250)

        self.RSPE=Entry()
        self.RREC=Entry()
        self.RPRE=Entry()
        self.RNPV=Entry()

        self.RSPE.place(x=1100, y=100)
        self.RREC.place(x=1100, y=150)
        self.RPRE.place(x=1100, y=200)
        self.RNPV.place(x=1100, y=250)

        self.RSPE.insert(END, str(SPE))
        self.RREC.insert(END, str(REC))
        self.RPRE.insert(END, str(PREC))
        self.RNPV.insert(END, str(NPV))

        self.resr.insert(END, str(result))
window=Tk()
mywin=MyWindow(window)
window.title('Calculadora riesgo restenosis tras ICP')
window.geometry("1300x650+10+10")

load1 = Image.open("USAL2.jpg")
render1 = ImageTk.PhotoImage(load1)
img1 = Label(window, image=render1)
img1.image = render1
img1.place(x=10, y=40)




window.mainloop()
