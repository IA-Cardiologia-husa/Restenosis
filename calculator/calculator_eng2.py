import pandas as pd
import numpy as np
import sklearn
import pickle
from tkinter import *
from PIL import ImageTk, Image
from funcs_bokeh_eng import *
from bokeh.io import export_png

class MyWindow:
    def __init__(self, win):
        self.lbl0=Label(win, text='Stent Restenosis calculator', font='Helvetica 14 bold')
        self.lbl1=Label(win, text='Diabetes', font='Helvetica 10 bold')
        self.lbl2=Label(win, text='>=2 vessel disease', font='Helvetica 10 bold')
        self.lbl3=Label(win, text='Post PCI thrombus', font='Helvetica 10 bold')
        self.lbl4=Label(win, text='Post PCI TIMI flow', font='Helvetica 10 bold')
        self.lbl5=Label(win, text='Abnormal total colesterol', font='Helvetica 10 bold')
        self.lbl6=Label(win, text='Abnormal platelets', font='Helvetica 10 bold')
        self.resl=Label(win, text='Restenosis score', font='Helvetica 10 bold')
        self.SPE=Label(win, text='Specifity', font='Helvetica 10 bold')
        self.REC=Label(win, text='Sensitivity', font='Helvetica 10 bold')
        self.PRE=Label(win, text='Precision', font='Helvetica 10 bold')
        self.NPV=Label(win, text='NPV', font='Helvetica 10 bold')

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

        rad1 = Radiobutton(window,text='Yes', variable=self.num1, value=1)
        rad2 = Radiobutton(window,text='No', variable=self.num1, value=0)
        rad1.place(x=165, y=180)
        rad2.place(x=215, y=180)
        rad3 = Radiobutton(window,text='Yes', variable=self.num2, value=1)
        rad4 = Radiobutton(window,text='No', variable=self.num2, value=0)
        rad3.place(x=165, y=220)
        rad4.place(x=215, y=220)


        rad50 = Radiobutton(window,text='0', variable=self.num3, value=0)
        rad51 = Radiobutton(window,text='1', variable=self.num3, value=1)
        rad52 = Radiobutton(window,text='2', variable=self.num3, value=2)
        rad53 = Radiobutton(window,text='3', variable=self.num3, value=3)
        rad54 = Radiobutton(window,text='4', variable=self.num3, value=4)
        rad55 = Radiobutton(window,text='5', variable=self.num3, value=5)
        rad50.place(x=165, y=270)
        rad51.place(x=195, y=270)
        rad52.place(x=225, y=270)
        rad53.place(x=255, y=270)
        rad54.place(x=285, y=270)
        rad55.place(x=315, y=270)


        rad7 = Radiobutton(window,text='1', variable=self.num4, value=1)
        rad8 = Radiobutton(window,text='2', variable=self.num4, value=2)
        rad81 = Radiobutton(window,text='3', variable=self.num4, value=3)
        rad7.place(x=165, y=320)
        rad8.place(x=215, y=320)
        rad81.place(x=265, y=320)
        rad9 = Radiobutton(window,text='Yes', variable=self.num5, value=1)
        rad10 = Radiobutton(window,text='No', variable=self.num5, value=0)
        rad9.place(x=165, y=370)
        rad10.place(x=215, y=370)
        rad11 = Radiobutton(window,text='Yes', variable=self.num6, value=1)
        rad12 = Radiobutton(window,text='No', variable=self.num6, value=0)
        rad11.place(x=165, y=420)
        rad12.place(x=215, y=420)
        self.resr=Entry()
        self.lbl0.place(x=15,y=5)
        self.lbl1.place(x=15, y=180)
        self.lbl2.place(x=15, y=220)
        self.lbl3.place(x=15, y=270)
        self.lbl4.place(x=15, y=320)
        self.lbl5.place(x=15, y=370)
        self.lbl6.place(x=15, y=420)
        self.b1=Button(win, text='Run model', command=self.add, font='Helvetica 14 bold')
        self.b1.place(x=15, y=470)
        self.resl.place(x=15, y=530)
        self.resr.place(x=150, y=530)
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
                            x_label = '1 - Specifity', y_label = 'Sensitivity',
                            x = fpr1, y = tpr1, x_p = 1-SPE, y_p = REC, colors = colors,
                            legend_pos = 'bottom_right',
                            plot_width=450, plot_height=330)
        fig2 = figure_bokeh('P-R', thresholds=thresholds_pr, x_label = 'Sensitivity',
                            y_label = 'Precision', x = rec1, y = prec1,
                            x_p = REC, y_p = PREC, colors = colors,
                            legend_pos = 'top_right',
                            plot_width=450, plot_height=330)
        # fig3 = figure_bokeh('NPV-SPE', thresholds=thresholds_npv,
        #                     x_label = 'Specifity',
        #                     y_label = 'Negative Predictive Value',
        #                     x = spe1, y = npv1, x_p = SPE, y_p = NPV,
        #                     colors = colors[::-1], legend_pos = 'bottom_right',
        #                     range_y = (0.85,1),
        #                     plot_width=350, plot_height=500)
        export_png(fig1, filename="fig1.png")
        export_png(fig2, filename="fig2.png")
        # export_png(fig3, filename="fig3.png")

        load2 = Image.open("fig1.png")
        render2 = ImageTk.PhotoImage(load2)
        img2 = Label(window, image=render2)
        img2.image = render2
        img2.place(x=350, y=10)

        load3 = Image.open("fig2.png")
        render3 = ImageTk.PhotoImage(load3)
        img3 = Label(window, image=render3)
        img3.image = render3
        img3.place(x=350, y=350)

        # load4 = Image.open("fig3.png")
        # render4 = ImageTk.PhotoImage(load4)
        # img4 = Label(window, image=render4)
        # img4.image = render4
        # img4.place(x=900, y=510)

        self.SPE.place(x=15, y=560)
        self.REC.place(x=15, y=590)
        self.PRE.place(x=15, y=620)
        self.NPV.place(x=15, y=650)

        self.RSPE=Entry()
        self.RREC=Entry()
        self.RPRE=Entry()
        self.RNPV=Entry()

        self.RSPE.place(x=150, y=560)
        self.RREC.place(x=150, y=590)
        self.RPRE.place(x=150, y=620)
        self.RNPV.place(x=150, y=650)

        self.RSPE.insert(END, str(SPE))
        self.RREC.insert(END, str(REC))
        self.RPRE.insert(END, str(PREC))
        self.RNPV.insert(END, str(NPV))

        self.resr.insert(END, str(result))
window=Tk()
mywin=MyWindow(window)
window.title('Stent Restenosis calculator')
window.geometry("820x690+10+10")

load1 = Image.open("Logo_USAL_Color_2012.png")
load1 = load1.resize((161,122), Image.ANTIALIAS)
render1 = ImageTk.PhotoImage(load1)
img1 = Label(window, image=render1)
img1.image = render1
img1.place(x=10, y=40)

window.mainloop()
