from model_zoo import *
import mkl
mkl.set_num_threads(1)

# Script to run train_split_nested with outer cv k-folds and n repetitions.
# Inner cv is done with k-1 folds with 1 repetition.
# Each k-fold is run on a processor core.

# Returns dataframes with generalization results over test and training subsets
# and scores obtained in the inner cv k-folds

folds = 5
repetitions = 1
threads = 2

if __name__ == '__main__':
    data = pd.read_excel('./data/clean_df.xls')
    X = data.copy().drop('label', axis = 1)
    y = data.loc[:,'label']

    # model_dict = model_zoo

    inner_rkf = StratifiedKFold(n_splits = folds-1, random_state= 0, shuffle= True)
    for model_dict in model_zoo:
        print('**** TRAINING ' + model_dict['clf_name'] + ' ****')
        res_out = pd.DataFrame(columns = ['idx', 'proba', 'GT', 'train_test', 'split', 'seed'])
        res_inn = pd.DataFrame(columns = ['inner_train', 'inner_test', 'model', 'variables', 'split', 'seed'])
        model = model_dict['clf_fun']
        param_grid = model_dict['param_grid']

        folds = 10
        pool = mp.Pool(threads)
        res = [pool.apply_async(k_fold_nested, args = (X, y, folds, seed, model, param_grid, inner_rkf))
                                for seed in range(repetitions)]
        pool.close()
        for p in res:
            df_out, df_inn = p.get()
            res_out = res_out.append(df_out)
            res_inn = res_inn.append(df_inn)
        res_out.to_csv('./results/out_' + model_dict['clf_name'] + '_' + str(folds) + '.csv')
        res_inn.to_csv('./results/in_' + model_dict['clf_name'] + '_' + str(folds) + '.csv', sep=';')
        print('**** ' + model_dict['clf_name'] + ' FINISHED **** \n')

    print('FINISHED')
