# -- coding:utf-8 --

from utils_features_selection import *


def create_table(cols=['乳酸脱氢酶', '淋巴细胞(%)', '超敏C反应蛋白']):
    data_df_unna, data_pre_df = data_preprocess()
    data_df_unna = data_df_unna.dropna(subset=cols, how='any')

    cols.append('Type2')
    Tets_Y = data_pre_df.reset_index()[['PATIENT_ID', '出院方式']].copy()
    Tets_Y = Tets_Y.rename(columns={'PATIENT_ID': 'ID', '出院方式': 'Y'})
    y_true = Tets_Y['Y'].values

    x_col = cols[:-1]
    y_col = cols[-1]
    x_np = data_df_unna[x_col].values
    y_np = data_df_unna[y_col].values
    x_test = data_pre_df[x_col].values
    X_train, X_val, y_train, y_val = train_test_split(x_np, y_np, test_size=0.3, random_state=6)
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=1,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(x_test)

    print(classification_report(y_true, pred_test))

if __name__ == '__main__':
    create_table()
