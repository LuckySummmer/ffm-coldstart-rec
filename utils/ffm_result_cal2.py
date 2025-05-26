import scipy.io as scio
import numpy as np
import pandas as pd
import csv
import os
from eval_metrics import precision_at_k, recall_at_k, ndcg_k, hitrate_at_k


def load_dict_from_csv(file_path):
    """
    从csv加载特征字典
    """
    df = pd.read_csv(file_path, index_col='FeatureKey')
    return df['FeatureID'].to_dict()



def parse_test_ffm_file(ffm_file, dict_file, num_test_users, num_items):

    with open(ffm_file, 'r') as f_actual:
        lines_actual = f_actual.readlines()


    start_idx = 0

    feature_dict = load_dict_from_csv(dict_file)
    reverse_dict = {v:k for k, v in feature_dict.items()}

    actual = []

    for _ in range(num_test_users):
        user_lines_actual = lines_actual[start_idx : start_idx+num_items]
        start_idx = start_idx + num_items


        recom_items_id = []


        for i, line1 in enumerate(user_lines_actual):

            parts = line1.strip().split()
            label = int(parts[0])
            all_field = [tuple(map(int, feat.split(':'))) for feat in parts[1:]]
            
            field1 = all_field[0]
            field2 = all_field[1]

            userid = field1[1]      # global only
            itemid = field2[1]      # global only
            user_ID = int(reverse_dict[userid][4:])     # local in user / actutal ID
            item_ID = int(reverse_dict[itemid][4:])
            # print(f"{label} {userid}-{user_ID} {itemid}-{item_ID}")

            if label == 1:
                recom_items_id.append(item_ID)
                # print(label, field1, field2)
        

        actual.append(recom_items_id)
    
    # print(actual[:10])
    return actual



def parse_output_file(output_file, num_test_users, num_items):

    with open(output_file, 'r') as f_predict:
            lines_predict = f_predict.readlines()
    
    predict = []
    start_idx = 0

    for _ in range(num_test_users):
        user_lines_actual = lines_predict[start_idx : start_idx+num_items]
        start_idx = start_idx + num_items

        user_pred = np.array([float(line.strip()) for line in user_lines_actual])
        pred_sorted_indices = user_pred.argsort()[::-1]

        predict.append(pred_sorted_indices)
    
    # print(predict[:10])
    return predict


def calculate_metrics(actual, predicted, topks):
    precision_record = []
    recall_record = []
    ndcg_record = []
    hitrate_record = []

    for k in topks:
        precision_record.append(precision_at_k(actual, predicted, k))
        recall_record.append(recall_at_k(actual, predicted, k))
        ndcg_record.append(ndcg_k(actual, predicted, k))
        hitrate_record.append(hitrate_at_k(actual, predicted, k))

    return precision_record, recall_record, ndcg_record, hitrate_record


# if __name__ == "__main__":
#     dataname = 'delicious_user'
#     # dataname = 'lastfm_user'
#     ffm_file = f'../dataset/{dataname}/{dataname}_test_0.ffm'

#     mat_file = f'../dataset/{dataname}/{dataname}.mat'
#     data = scio.loadmat(mat_file)
#     user_item = data['user_item'].tocoo().tocsr()
#     num_users, num_items = user_item.shape
#     train_indices = data['train'][0]
#     test_indices = data['test'][0]
#     num_train_users = train_indices.shape[0]
#     num_test_users = test_indices.shape[0]
#     print(num_test_users, num_train_users, num_users, num_items)

#     dict_file = f'../dataset/{dataname}/{dataname}_all_feature_dict.csv'
#     actual = parse_test_ffm_file(ffm_file, dict_file, num_test_users, num_items)
#     topks=[5, 10, 15, 20]
#     csv_file = f'../results/tune_ffm_{dataname}_0_t50_val.csv'

#     print(actual)

#     # file_exists = os.path.isfile(csv_file)

#     # find all output_files
#     output_files = []
#     for k in [4, 8, 12, 16]:
#         for r in [0.01, 0.05, 0.1]:
#             for l in ['0.00001', '0.00002', '0.00005', '0.0001']:
#                 o_f = f'../outputs/{dataname}/output_{dataname}_k{k}_r{r}_l{l}_t50_0.txt'
#                 output_files.append(o_f)
#                 predict = parse_output_file(o_f, num_test_users, num_items)

#                 print(predict[0][0], end=' ')

#                 precision_record, recall_record, ndcg_record, hitrate_record = calculate_metrics(actual, predict, topks)
                
#                 with open(csv_file, 'a+', newline='') as f_csv:
#                     writer = csv.writer(f_csv)
#                     if k == 4 and r == 0.01 and l == '0.00001':
#                         p_lst = ['p' + str(i) for i in topks]
#                         r_lst = ['r' + str(i) for i in topks]
#                         n_lst = ['n' + str(i) for i in topks]
#                         h_lst = ['h' + str(i) for i in topks]
#                         header = ['k', 'r', 'l'] + p_lst + r_lst + n_lst + h_lst
#                         writer.writerow(header)
                    
#                     row = [k, r, l] + precision_record + recall_record + ndcg_record + hitrate_record
#                     writer.writerow(row)
