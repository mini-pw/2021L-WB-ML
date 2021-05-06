import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")


def preprocess_time_series():
    data = pd.read_excel('data/time_series_375_prerpocess.xlsx', index_col=[0, 1])
    data = data.dropna(thresh=6)
    data.to_parquet('data/time_series_375.parquet')
    data = pd.read_excel('data/time_series_test_110_preprocess.xlsx', index_col=[0, 1])
    data.to_parquet('data/time_series_test_110.parquet')

def decision_tree(x: pd.Series):
    if x['乳酸脱氢酶'] >= 365:
        return 1

    if x['超敏C反应蛋白'] < 41.2:
        return 0

    if x['淋巴细胞(%)'] > 14.7:
        return 0
    else:
        return 1


def compute_f1_all(data, features, days=10):
    day_list = list(range(0, days + 1))
    sample_num = []
    survival_num = []
    death_num = []
    f1 = []
    precision = []
    recall = []
    add_before_f1 = []
    for i in range(0, days + 1):
        if i == 0:
            data_subset = data.loc[data['t_diff'] <= 0].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= 0]
        else:
            data_subset = data.loc[data['t_diff'] == i].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= i]
        if data_subset.shape[0] > 0:
            sample_num.append(data_subset.shape[0])
            survival_num.append(sum(data_subset['出院方式'] == 0))
            death_num.append(sum(data_subset['出院方式'] == 1))
            pred = data_subset[features].apply(decision_tree, axis=1)

            f1.append(f1_score(data_subset['出院方式'].values, pred, average='macro'))
            precision.append(precision_score(data_subset['出院方式'].values, pred))

            recall.append(recall_score(data_subset['出院方式'].values, pred))

            add_before_f1.append(f1_score(data_subset_sum['出院方式'].values,
                                          data_subset_sum[features].apply(decision_tree, axis=1),
                                          average='macro'))

        else:
            sample_num.append(np.nan)
            survival_num.append(np.nan)
            death_num.append(np.nan)
            f1.append(np.nan)
            precision.append(np.nan)
            recall.append(np.nan)
            add_before_f1.append(np.nan)
    return day_list, f1, precision, recall, sample_num, survival_num, death_num, add_before_f1




def plot_f1_time_single_tree(data, features, path='f1_score_time.png'):
    test_model_result = pd.DataFrame()
    test_model_result['day'], test_model_result['f1-score'], test_model_result['precision-score'], \
    test_model_result['recall-score'], test_model_result['sample_num'], test_model_result['survival_num'], \
    test_model_result['death_num'], test_model_result['add_before_f1'] = compute_f1_all(data, features, days=18)
    fig = plt.figure(figsize=(8, 6))
    plt.tick_params(labelsize=20)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    n1 = ax.bar(test_model_result['day'], test_model_result['sample_num'], label='Death', color='red', alpha=0.5
                , zorder=0)
    n2 = ax.bar(test_model_result['day'], test_model_result['survival_num'], label='Survival', color='lightgreen',
                alpha=1, zorder=5)
    p1 = ax2.plot(test_model_result['day'], test_model_result['f1-score'], marker='o', linestyle='-', color='black',
                  label='f1 score', zorder=10)

    p2 = ax2.plot(test_model_result['day'], test_model_result['add_before_f1'], marker='o', linestyle='-', color='blue',
                  label='cumulative f1 score', zorder=10)

    fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.95), bbox_transform=ax.transAxes, fontsize=14)

    ax.set_xlabel('days to outcome', fontsize=20)
    ax2.set_ylabel('f1-score(macro avg)', fontsize=20)
    ax.set_ylabel('sample_num', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xticks(list(range(0, 19, 2)))
    plt.savefig(path, dpi=500, bbox_inches='tight')
    plt.show()


def merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=None, time_form='diff'):
    data = data.reset_index(level=1)
    t_diff = data['出院时间'].dt.normalize() - data['RE_DATE'].dt.normalize()
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


def plot_f1_time_single_tree_test_train(plotnum="3d"):
    features = ['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)']
    data1 = pd.read_parquet('data/time_series_375.parquet')[['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)', '出院时间', '出院方式']]
    data2 = pd.read_parquet('data/time_series_test_110.parquet')[['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)', '出院时间', '出院方式']]
    data = data1.append(data2)

    data = merge_data_by_sliding_window(data, n_days=1, dropna=True, time_form='diff')
    data = data.sort_index(level=(0, 1), ascending=True)
    data = data.reset_index()
    data = data.dropna(how='all', subset=['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)'])

    data2 = merge_data_by_sliding_window(data2, n_days=1, dropna=True, time_form='diff')
    data2 = data2.sort_index(level=(0, 1), ascending=True)
    data2 = data2.reset_index()
    data2 = data2.dropna(how='all', subset=['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)'])

    if plotnum == "3d":
        plot_f1_time_single_tree(data, features, path='f1_time_train_test.png')
    elif plotnum == "3e":
        plot_f1_time_single_tree(data2, features, path='f1_time_test.png')
    else:
        print("wrong figure code given")

def show_fig_3d():
    preprocess_time_series()
    plot_f1_time_single_tree_test_train(plotnum="3d")
def show_fig_3e():
    preprocess_time_series()
    plot_f1_time_single_tree_test_train(plotnum="3e")

if __name__ == '__main__':
    show_fig_3d()
    show_fig_3e()