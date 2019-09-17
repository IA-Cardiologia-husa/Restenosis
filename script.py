import libreria
from libreria import *
import multiprocessing as mp
import pandas as pd
import numpy as np

data = pd.read_excel('./data/df_limpio.xls')

vars_v5_restenosis = ['stent', 'tirofiban', 'sexo', 'diabetes', 'hipertension', 'his_familiar', 'edad', 'pci previa',
                      'angina', 'medhis.cignumb', 'medhis.ca', 'medhis.alcnumb', 'medhis.imp', 'medhis.col',
                      'tabaquismo_2', 'tabaquismo_3',
                      'ostial_segment',
                      'angio.illvess2', 'angio.illvess3',
                      'localiz_LCX', 'localiz_RCA', 'localiz_RDPA',
                      'obs_length_pre', 'perc_diam_pre', 'perc_area_sten_pre', 'thrombus_pre', 'obs_diam_pre',
                      'perc_diam_post', 'perc_area_sten_post', 'obs_diam_post', 'acute_gain',                     
                      'pci_result_post', 'no_reflow_post', 'thrombus_post',
                      'l_type_pre_B1', 'l_type_pre_B2', 'l_type_pre_C', 'l_type_pre_NS',
                      'timi_basal', 'tmpg_basal', 'timi_post', 'tmpg_post',
                      'treat.stenttot', 'treat.2', 'treat.3',
                      'lab.cho.a', 'lab.cpk.a', 'lab.cpk-mb.a', 'lab.crt.a', 'lab.hbn.a', 'lab.hct.a', 'lab.ldl.a',
                      'lab.plt.a', 'lab.tropi.a', 'lab.tropt.a', 'lab.wbc.a',
                      'lab.cho', 'lab.cpk', 'lab.cpk-mb', 'lab.crt', 'lab.hbn', 'lab.hct', 'lab.ldl', 'lab.plt',
                      'lab.wbc',
                      'patvis.1.1', 'patvis.1.2', 'patvis.1.3', 'patvis.1.4', 'patvis.1.5', 'patvis.1.6', 'patvis.1.7']
vars_v6_restenosis = vars_v5_restenosis + ['vitals.sbp', 'vitals.dbp', 'vitals.pulse', 'vitals.wt', 'vitals.ht']

X = data[vars_v6_restenosis].copy()
y = data['reestenosis']

rf_dict = {'clf_name': 'RF', 'clf_fun': RandomForestClassifier(n_estimators=1000, max_depth=5), 'param_grid': {}}
et_dict = {'clf_name': 'ET', 'clf_fun': ExtraTreesClassifier(n_estimators=1000, max_depth=5, random_state = 0), 'param_grid': {}}
lr_dict = {'clf_name': 'LR', 'clf_fun': LogisticRegression(penalty = 'none', solver = 'newton-cg', random_state = 0), 'param_grid': {}}
PCA_et_dict = {'clf_name': 'PCA_ET',
               'clf_fun': Pipeline(steps = [('pca', PCA(n_components=5)),
                                            ('etr', ExtraTreesClassifier(n_estimators=500, max_depth=2, random_state = 0))]),
               'param_grid': {}}


anova_dict = {'fs_name': 'anova', 'fs_fun': sel_anova, 'param_grid':{'k': 5}}
sfm_dict = {'fs_name': 'sfm', 'fs_fun': sel_feature_importances, 'param_grid':{'k':5}}
no_sel_dict = {'fs_name': 'no_sel', 'fs_fun': no_sel, 'param_grid':{'k':5}}

if __name__ == '__main__':
    fs_dict = no_sel_dict
    clf_dict = PCA_et_dict
    results = pd.DataFrame(columns = ['idx', 'proba', 'GT', 'train_test', 'split', 'seed'])
    folds = 10
    pool = mp.Pool(4)
    
    t = time.time()
    res = [pool.apply_async(k_fold, args = (X, y, folds, seed, no_preproc, fs_dict, clf_dict, no_hyper_tun)) for seed in range(10)]   
    pool.close()
    for p in res:
        df = p.get()
        results = results.append(df)
    print(time.time()-t)
    results.to_csv(fs_dict['fs_name'] + '_' + clf_dict['clf_name'] + '_' + str(folds) + '.csv')
    print('\n')
    
    print(roc_auc_score(results.loc[(results.train_test == 1), 'GT'].astype(np.int), results.loc[(results.train_test == 1), 'proba']))

    print("FIN")
