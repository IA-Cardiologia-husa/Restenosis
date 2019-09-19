from lib_ml import *

clf1 = LogisticRegression(random_state=0, penalty = 'l2', n_jobs = 1)
model1 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', clf1)])
param_grid1 = {'fs__k': [2,4], 'clf__C': [0.1,1]}
model_dict1 = {'clf_name': 'median_standar_anova_lr',
               'clf_fun': model1 ,
               'param_grid': param_grid1}

## LOGISTIC REGRESSION MODELS
model_lr = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', LogisticRegression(random_state=0, penalty = 'l2', n_jobs = 1))])
param_grid_lr = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__C': [0.1,1,10,100,1000]}
model_dict_lr = {'clf_name': 'median_standar_anova_lr', 'clf_fun': model_lr ,'param_grid': param_grid_lr}

model_lr2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2))),
                  ('clf', LogisticRegression(random_state=0, penalty = 'l2', n_jobs = 1))])
param_grid_lr2 = {'fs__max_features': [2,4,6,8,10,12,14,16,18,20], 'clf__C': [0.1,1,10,100,1000]}
model_dict_lr2 = {'clf_name': 'median_standar_ef_lr', 'clf_fun': model_lr2 ,'param_grid': param_grid_lr2}

model_lr_noreg = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', LogisticRegression(random_state=0, penalty = 'none', solver = 'newton-cg', n_jobs = 1))])
param_grid_lr = {'fs__k': [2,4,6,8,10,12,14,16,18,20]}
model_dict_lr = {'clf_name': 'median_standar_anova_lr_noreg', 'clf_fun': model_lr_noreg ,
                 'param_grid': param_grid_lr_noreg}

model_lr_noreg2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2))),
                  ('clf', LogisticRegression(random_state=0, penalty = 'none', solver = 'newton-cg', n_jobs = 1))])
param_grid_lr_noreg2 = {'fs__max_features': [2,4,6,8,10,12,14,16,18,20]}
model_dict_lr_noreg2 = {'clf_name': 'median_standar_ef_lr_noreg', 'clf_fun': model_lr_noreg2 ,
                        'param_grid': param_grid_lr_noreg2}

# RANDOM FOREST MODELS
model_rf = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf',  RandomForestClassifier(random_state=0, n_jobs = 1, n_estimators = 500, max_features = 1))])
param_grid_rf = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_rf = {'clf_name': 'median_anova_rf', 'clf_fun': model_rf ,'param_grid': param_grid_rf}

model_rf2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                   ('fs', SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2))),
                   ('clf', RandomForestClassifier(random_state=0, n_jobs = 1, n_estimators = 500, max_features = 1))])
param_grid_rf2 = {'fs__max_features': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_rf2 = {'clf_name': 'median_ef_rf', 'clf_fun': model_rf2 ,'param_grid': param_grid_rf2}

# EXTREMELY RANDOMIZED TREES MODELS
model_ef = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', ExtraTreesClassifier(random_state = 0, n_jobs = 1, n_estimators = 500,
                                               max_features = 1))])
param_grid_ef = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_ef = {'clf_name': 'median_anova_ef', 'clf_fun': model_ef ,'param_grid': param_grid_ef}


model_ef2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('fs', SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2)),
                  ('clf', ExtraTreesClassifier(random_state = 0, n_jobs = 1, n_estimators = 500,
                                               max_features = 1))])
param_grid_ef2 = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_ef2 = {'clf_name': 'median_ef_ef', 'clf_fun': model_ef2 ,'param_grid': param_grid_ef2}

# GRADIENT BOOSTING MODELS
model_gb = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', GradientBoostingClassifier(random_state = 0, n_estimators = 500, max_features = 1))])
param_grid_gb = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_gb = {'clf_name': 'median_anova_gb', 'clf_fun': model_gb ,'param_grid': param_grid_gb}

model_gb2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('fs', SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2)),
                  ('clf', GradientBoostingClassifier(random_state = 0, n_estimators = 500, max_features = 1))])
param_grid_gb2 = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__max_depth': [2,3,4,5,10]}
model_dict_gb2 = {'clf_name': 'median_ef_gb', 'clf_fun': model_gb2 ,'param_grid': param_grid_gb2}

# SUPPORT VECTOR MACHINES MODELS
model_sv = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs', SelectKBest(score_func=f_classif)),
                  ('clf', SVC(random_state = 0, kernel = 'rbf', probability = True))])
param_grid_sv = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__C': [0.1,1,10,100,1000],
                'clf__gamma': [0.1, 1, 10, 100]}
model_dict_sv = {'clf_name': 'median_escaler_anova_svc', 'clf_fun': model_sv ,'param_grid': param_grid_sv}

model_sv2 = Pipeline([('mv', preprocessing.Imputer(strategy='median')),
                  ('scaler', preprocessing.StandardScaler()),
                  ('fs',SelectFromModel(threshold=-np.inf, estimator = ExtraTreesClassifier(n_estimators = 50, random_state = 0, max_features = 1, max_depth = 2)),
                  ('clf', SVC(random_state = 0, kernel = 'rbf', probability = True))])
param_grid_sv2 = {'fs__k': [2,4,6,8,10,12,14,16,18,20], 'clf__C': [0.1,1,10,100,1000],
                'clf__gamma': [0.1, 1, 10, 100]}
model_dict_sv2 = {'clf_name': 'median_escaler_ef_svc', 'clf_fun': model_sv2 ,'param_grid': param_grid_sv2}



model_zoo = [model_dict_lr, model_dict_lr2, model_dict_lr_noreg, model_dict_lr_noreg2,
             model_dict_rf, model_dict_rf2, model_dict_ef, model_dict_ef2,
             model_dict_gb, model_dict_gb2, model_dict_sv, model_dict_sv2]
