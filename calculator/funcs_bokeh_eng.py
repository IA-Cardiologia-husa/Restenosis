import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve
import pickle

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models.markers import CircleX
from bokeh.util.browser import view
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import column, gridplot
from bokeh.models import Circle, ColumnDataSource, Div, Grid, Line, LinearAxis, Plot, Range1d, HoverTool
from bokeh.resources import INLINE

def pred_metric(y_prob,y_resp):

    y_ordered = [x for _,x in sorted(zip(y_prob,y_resp))]
    y_prob_ordered = [x for x,_ in sorted(zip(y_prob,y_resp))]
    sens = []
    spec = []
    prec = []
    nprv = []
    for i in range(len(y_ordered)):
        tn = y_ordered[0:i].count(0)
        fn = y_ordered[0:i].count(1)
        tp = y_ordered[i:].count(1)
        fp = y_ordered[i:].count(0)
#         print(i, tn, fn, tp, tp)
        if((tp+fn)!=0):
            sens.append(tp/(tp+fn))
        else:
            sens.append(1)

        if((tn+fp)!=0):
            spec.append(tn/(tn+fp))
        else:
            spec.append(1)

        if((tp+fp)!=0):
            prec.append(tp/(tp+fp))
        else:
            prec.append(1)

        if((tn+fn)!=0):
            nprv.append(tn/(tn+fn))
        else:
            nprv.append(1)

    return np.array(prec), np.array(sens), np.array(spec), np.array(nprv)


def point_scores(data, proba):
    cm = confusion_matrix(data.loc[:, 'GT'],
                          data.loc[:, 'proba'] >= proba) * 100 / 5260
    TN, TP, FN, FP = cm[0,0], cm[1,1], cm[1,0], cm[0,1]
    PREC = TP / (TP + FP)
    REC = TP / (TP + FN)
    SPE = TN / (TN + FP)
    NPV = TN / (TN + FN)
    return PREC, REC, SPE, NPV


def gracia_curves(data):
    fpr1, tpr1, _ = roc_curve(data['GT'],data['proba'])
    prec1, rec1, _ = precision_recall_curve(data['GT'],data['proba'])
    _, _, spe1, npv1 = pred_metric(data['proba'], data['GT'])
    return fpr1, tpr1, prec1, rec1, spe1, npv1


def figure_bokeh(curva, thresholds, x_label, y_label,x, y, x_p, y_p, colors, legend_pos,
                range_x = (0,1), range_y=(0,1), plot_width=700, plot_height=550):
    patchx = []
    patchy_el = [0, 0 ,1, 1]
    patchy =  []
    alphas = [0.3, 0.3, 0.3, 0.3]
    for i in range(1, len(thresholds), 1):
        patchx = patchx + [[thresholds[i-1], thresholds[i], thresholds[i], thresholds[i-1]]]
        patchy = patchy + [patchy_el]

    p = figure(plot_width=plot_width, plot_height=plot_height,
               title="POINT OVER THE " + curva + " CURVE",
               x_axis_label=x_label , y_axis_label = y_label,
               x_range=range_x, y_range=range_y, toolbar_location=None, tools = "")

    for i in range(len(patchx)):
        p.patch(patchx[i], patchy[i], fill_color = colors[i], alpha = alphas[i], line_width = 0)

    linea = p.line(x, y, line_width = 3, legend = "ERT MODEL", name = 'CURVE')
    punto = p.x(x_p, y_p, size = 10, line_width = 6,  alpha = 1, color = 'black',
                legend = 'POINT: ' + x_label + ' =  %0.2f; ' % x_p + y_label + ' = %0.2f' % y_p,
                name = 'POINT')
    p.add_tools(HoverTool(tooltips=[(x_label, '@x'), (y_label, '@y')], renderers = [linea], mode='vline'))

    p.legend.location = legend_pos
    p.legend.border_line_width = 3
    p.legend.border_line_color = "navy"
    p.legend.border_line_alpha = 0.5

    return p
