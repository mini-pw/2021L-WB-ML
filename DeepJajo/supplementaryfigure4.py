import pandas as pd
import numpy as np

from utils_features_selection import *

from xgboost import plot_tree

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def features_selection():
    X_data_all_features, Y_data, x_col = data_read_and_split()
    import_feature = pd.DataFrame()
    import_feature['col'] = x_col
    import_feature['xgb'] = 0
    # 重复100次试验
    for i in range(100):  # 50,150
        x_train, x_test, y_train, y_test = train_test_split(X_data_all_features, Y_data, test_size=0.3, random_state=i)
        model = xgb.XGBClassifier(
            max_depth=4
            , learning_rate=0.2
            , reg_lambda=1
            , n_estimators=150
            , subsample=0.9
            , colsample_bytree=0.9)
        model.fit(x_train, y_train)
        import_feature['xgb'] = import_feature['xgb'] + model.feature_importances_ / 100
    import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
    print('Top 10 features:')
    print(import_feature.head(10))
    indices = np.argsort(import_feature['xgb'].values)[::-1]
    Num_f = 10
    indices = indices[:Num_f]

    plt.subplots(figsize=(12, 10))
    g = sns.barplot(y=import_feature.iloc[:Num_f]['col'].values[indices],
                    x=import_feature.iloc[:Num_f]['xgb'].values[indices],
                    orient='h')  # import_feature.iloc[:Num_f]['col'].values[indices]
    g.set_xlabel("Relative importance", fontsize=18)
    g.set_ylabel("Features", fontsize=18)
    g.tick_params(labelsize=14)
    sns.despine()
    plt.show()
    import_feature_cols = import_feature['col'].values[:10]

    num_i = 1
    val_score_old = 0
    val_score_new = 0
    while val_score_new >= val_score_old:
        val_score_old = val_score_new
        x_col = import_feature_cols[:num_i]
        print(x_col)
        X_data = X_data_all_features[x_col]  # .values
        print('5-Fold CV:')
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,
                                                                                                Y_data.values)
        print("Train AUC-score is %.4f ; Validation AUC-score is %.4f" % (acc_train, acc_val))
        print("Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f" % (acc_train_std, acc_val_std))
        val_score_new = acc_val
        num_i += 1

    print('Selected features:', x_col[:-1])

    return list(x_col[:-1])

def Compare_with_other_method(sub_cols=['乳酸脱氢酶', '淋巴细胞(%)', '超敏C反应蛋白']):
    x_np, y_np, x_col = data_read_and_split(is_dropna=True, sub_cols=sub_cols)

    X_train, X_val, y_train, y_val = train_test_split(x_np, y_np, test_size=0.3, random_state=6)

    xgb_n_clf = xgb.XGBClassifier(
        max_depth=4
        , learning_rate=0.2
        , reg_lambda=1
        , n_estimators=150
        , subsample=0.9
        , colsample_bytree=0.9
        , random_state=0)
    tree_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4)  # random_state=0,之前没加
    RF_clf1 = RandomForestClassifier(random_state=0, n_estimators=150, max_depth=4, )
    LR_clf = linear_model.LogisticRegression(random_state=0, C=1, solver='lbfgs')
    LR_reg_clf = linear_model.LogisticRegression(random_state=0, C=0.1, solver='lbfgs')

    fig = plt.figure(dpi=400, figsize=(16, 8))

    Num_iter = 100

    i = 0
    labels_names = []
    Moodel_name = ['Multi-tree XGBoost with all features',
                   'Decision tree with all features',
                   'Random Forest with all features',
                   'Logistic regression with all features with regularization parameter = 1 (by default)',
                   'Logistic regression with all features with regularization parameter = 10', ]
    for model in [xgb_n_clf, tree_clf, RF_clf1, LR_clf, LR_reg_clf]:
        print('Model:' + Moodel_name[i])
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np.values, y_np.values, Num_iter, model,
                                                                              score_type='f1')
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np.values, y_np.values, Num_iter, model,
                                                                              score_type='auc')
        print('AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f ' % (
        acc_train, acc_train_std, acc_val, acc_val_std))

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]
        pred_val_probe = model.predict_proba(X_val)[:, 1]
        plot_roc(y_train, pred_train_probe, Moodel_name[i], fig, labels_names, i)  # 为了画si图4 train
        print('AUC socre:', roc_auc_score(y_val, pred_val_probe))

        i = i + 1

    x_np_sel = x_np[sub_cols]  # 选择三特征
    X_train, X_val, y_train, y_val = train_test_split(x_np_sel, y_np, test_size=0.3, random_state=6)

    # 为了三特征的模型对比
    xgb_clf = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=1,
        random_state=0,
    )

    tree_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
    RF_clf2 = RandomForestClassifier(random_state=0, n_estimators=1, max_depth=3, )

    # i = 0
    Moodel_name = ['Single-tree XGBoost with three features',
                   'Decision tree with three features',
                   'Random Forest with a single tree constraint with three features', ]
    for model in [xgb_clf, tree_clf, RF_clf2]:
        print('Model' + Moodel_name[i - 5])
        # f1的结果
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np_sel.values, y_np.values, Num_iter,
                                                                              model, score_type='f1')
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np_sel.values, y_np.values, Num_iter,
                                                                              model, score_type='auc')
        print('AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f ' % (
        acc_train, acc_train_std, acc_val, acc_val_std))

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]  # 为了画si图4中的train
        pred_val_probe = model.predict_proba(X_val)[:, 1]  # 为了画si图4中的test
        plot_roc(y_train, pred_train_probe, Moodel_name[i - 5], fig, labels_names, i)  # 为了画si图4中的train
        print('AUC socre:', roc_auc_score(y_val, pred_val_probe))

        i = i + 1

    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='SouthEastOutside', fontsize=14)
    plt.savefig('AUC_train.png')
    plt.show()

