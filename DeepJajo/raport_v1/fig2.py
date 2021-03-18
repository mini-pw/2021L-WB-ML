import pandas as pd
import numpy as np

from utils_features_selection import *

from xgboost import plot_tree

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def show_fig(cols=['乳酸脱氢酶', '淋巴细胞(%)', '超敏C反应蛋白']):
    print('single_tree:\n')
    # 获取375病人（data_df_unna） 和110病人（data_pre_df）数据
    data_df_unna, data_pre_df = data_preprocess()
    # 去掉全空行，此时375总数目变成351
    data_df_unna = data_df_unna.dropna(subset=cols, how='any')

    cols.append('Type2')
    # 获取病人的结局标签
    Tets_Y = data_pre_df.reset_index()[['PATIENT_ID', '出院方式']].copy()
    # 修改dataframe的名字
    Tets_Y = Tets_Y.rename(columns={'PATIENT_ID': 'ID', '出院方式': 'Y'})
    # 获取110病人的标签数据
    y_true = Tets_Y['Y'].values

    x_col = cols[:-1]
    y_col = cols[-1]
    # 获取351病人的三特征数据
    x_np = data_df_unna[x_col].values
    # 获取351病人的标签数据
    y_np = data_df_unna[y_col].values
    # 获取110病人的三特征数据
    x_test = data_pre_df[x_col].values
    # 在351病人上划分训练集和验证集，此时110视为测试集
    X_train, X_val, y_train, y_val = train_test_split(x_np, y_np, test_size=0.3, random_state=6)
    # 限定单树xgb模型
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=1,
    )
    model.fit(X_train, y_train)

    # 训练集混淆矩阵
    pred_train = model.predict(X_train)
    show_confusion_matrix(y_train, pred_train)
    print(classification_report(y_train, pred_train))

    # 验证集混淆矩阵
    pred_val = model.predict(X_val)
    show_confusion_matrix(y_val, pred_val)
    print(classification_report(y_val, pred_val))
    # 测试集混淆矩阵

    pred_test = model.predict(x_test)
    print('True test label:', y_true)
    print('Predict test label:', pred_test.astype('int32'))
    show_confusion_matrix(y_true, pred_test)
    print(classification_report(y_true, pred_test))

    plt.figure(dpi=300, figsize=(8, 6))
    plot_tree(model)
    plt.show()

    graph = xgb.to_graphviz(model)
    graph.render(filename='single-tree.dot')

