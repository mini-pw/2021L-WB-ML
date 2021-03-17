import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score


def data_read_and_split(is_dropna=False,sub_cols=None):
    # data_df_unna为375数据集，data_pre_df为110数据集
    data_df_unna,data_pre_df = data_preprocess()
    if is_dropna==True:
        data_df_unna = data_df_unna.dropna(subset=sub_cols,how='any')
    col_miss_data = col_miss(data_df_unna)
    col_miss_data['Missing_part'] = col_miss_data['missing_count']/len(data_df_unna)
    sel_cols = col_miss_data[col_miss_data['Missing_part']<=0.2]['col']
    data_df_sel = data_df_unna[sel_cols].copy()
    cols = list(data_df_sel.columns)

    cols.remove('age') 
    cols.remove('gender')
    
    cols.remove('Type2')
    cols.append('Type2')
    data_df_sel2 = data_df_sel[cols]
    data_df_unna = pd.DataFrame()
    data_df_unna = data_df_sel2

    data_df_unna = data_df_unna.fillna(-1)

    x_col = cols[:-1]
    y_col = cols[-1]
    X_data = data_df_unna[x_col]
    Y_data = data_df_unna[y_col]
    return X_data,Y_data,x_col

def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

def data_preprocess():
    path_train_en = './data/time_series_375_prerpocess_en.xlsx' 
    data_df_unna_en = read_train_data(path_train_en)

    data_pre_df_en = pd.read_excel('./data/time_series_test_110_preprocess_en.xlsx', index_col=[0, 1])
    top3_feats_cols = ['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte']
    data_pre_df_en = merge_data_by_sliding_window(data_pre_df_en, n_days=1, dropna=True, subset=top3_feats_cols,
                                                     time_form='diff')
    data_pre_df_en = data_pre_df_en.groupby('PATIENT_ID').first().reset_index()
    data_pre_df_en = data_pre_df_en.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_pre_df_en = data_pre_df_en.drop_duplicates()
    return data_df_unna_en, data_pre_df_en

def is_number(s):
    if s is None:
        s = np.nan
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def read_train_data(path_train):
    data_df_en = pd.read_excel(path_train, index_col=[0, 1])  
    data_df_en = data_df_en.groupby('PATIENT_ID').last()

    lable = data_df_en['outcome'].values
    data_df_en = data_df_en.drop(['outcome', 'Admission time', 'Discharge time'], axis=1)
    data_df_en['Type2'] = lable
    data_df_en = data_df_en.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df_en = data_df_en.applymap(lambda x: x if is_number(x) else -1)
    data_df_en = data_df_en.astype(float)

    return data_df_en

def merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=None, time_form='diff'):
    data = data.reset_index(level=1)
    t_diff = data['Discharge time'].dt.normalize() - data['RE_DATE'].dt.normalize()
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    data = data.set_index('t_diff', append=True)
    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    )
    if dropna:
        data = data.dropna(subset=subset)         
    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])
    return data

def StratifiedKFold_func_with_features_sel(x, y,Num_iter=100,score_type = 'auc'):
    acc_v = []
    acc_t = []
    for i in range(Num_iter):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]
            model = xgb.XGBClassifier(max_depth=4,learning_rate=0.2,reg_alpha=1)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)
            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_te, pred))
                acc_t.append(roc_auc_score(y_tr, train_pred))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))    
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]

def supplementary_figure_1():
    X_data_all_features,Y_data,x_col = data_read_and_split()
    import_feature = pd.DataFrame()
    import_feature['col'] = x_col
    import_feature['xgb'] = 0
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(X_data_all_features, Y_data, test_size=0.3, random_state=i)
        
        ### ‘Multi-tree XGBoost algorithm’
        model = xgb.XGBClassifier(
                max_depth=4
                ,learning_rate=0.2
                ,reg_lambda=1
                ,n_estimators=150
                ,subsample = 0.9
                ,colsample_bytree = 0.9)
        model.fit(x_train, y_train)
        
        import_feature['xgb'] = import_feature['xgb']+model.feature_importances_/100
    import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
    return import_feature

def show_supplementary_figure_1():
    import_feature = supplementary_figure_1()
    # Sort feature importances from GBC model trained earlier
    indices = np.argsort(import_feature['xgb'].values)[::-1]
    Num_f = 10
    indices = indices[:Num_f]
    # Visualise these with a barplot
    plt.subplots(figsize=(16, 10))
    g = sns.barplot(y=import_feature.iloc[:Num_f]['col'].values[indices], x = import_feature.iloc[:Num_f]['xgb'].values[indices], orient='h') #import_feature.iloc[:Num_f]['col'].values[indices]
    g.set_xlabel("Relative importance",fontsize=18)
    g.set_ylabel("Features",fontsize=14)
    g.tick_params(labelsize=14)
    sns.despine() 
    plt.show()

def show_table_3():
    feature_sorted_by_importance = supplementary_figure_1()
    X_data_all_features,Y_data,x_col = data_read_and_split()
    import_feature_cols= feature_sorted_by_importance['col'].values[:10]
    num_i = 1
    val_score_old = 0
    val_score_new = 0
    results = pd.DataFrame(columns=["Features","AUC train", "AUC train std", "AUC validation", "AUC validation std"])
    while val_score_new >= val_score_old:
        val_score_old = val_score_new
        x_col = import_feature_cols[:num_i]
        X_data = X_data_all_features[x_col]
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
        results = results.append({"Features":x_col,
                        "AUC train":acc_train,
                         "AUC train std":acc_train_std,
                         "AUC validation":acc_val,
                         "AUC validation std":acc_val_std},
                         ignore_index=True)
        val_score_new = acc_val
        num_i += 1
    print(results)


if __name__ == '__main__':
    show_supplementary_figure_1()
    show_table_3()
