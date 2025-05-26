import argparse
import scipy.io as scio

import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
import utils.ffm_result_cal2 as ffm_cal


def load_config(config_path):
    """从配置文件加载超参数"""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def main():
    parser = argparse.ArgumentParser("Evaluate libffm with our dataset")
    parser.add_argument('-d', '--dataname', type=str, default='delicious_user',
                        choices=['delicious_user', 'lastfm_user', 'flickr', 'epinion', 'blog'],
                        help='dataname')
    parser.add_argument('-i', '--input', type=str, default='../data/processed',
                        help='Path to directory of the test ffm format data (To get labels)')
    parser.add_argument('-p', '--predict', type=str, default='../results/outputs',
                        help='Path to directory of the output of predict_ffm (To get predictions)')
    parser.add_argument('-r', '--result', type=str, default='../results/results_run',
                        help='Path to directory of the result save file')
    parser.add_argument('-c', '--config', type=str, default='../config',
                        help='Path to hyperparameter configuration file')
    parser.add_argument('-t', '--trials', action='store_true',
                        help='preprocess data of 10 trials')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        choices=[0, 1, 2],
                        help='indicator for result save file [0 -- no validation] [1 - with validation] [2 - with validation and auto-stop]')
    

    args = parser.parse_args()
    dataname = args.dataname
    actual_dir = f'{args.input}/{dataname}'
    predict_dir = f'{args.predict}/{dataname}'
    result_dir = f'{args.result}'


    config_path = f'{args.config}/params_{dataname}.txt'
    config = load_config(config_path)
    k_values = list(map(int, config['k_values'].split(',')))
    r_values = list(map(float, config['r_values'].split(',')))
    l_values = list(map(float, config['l_values'].split(',')))
    data_sets = config['data_sets'].split(',')
    assert data_sets[0] == dataname

    if args.trials:
        trials = list(range(10))
        if dataname == "delicious_user":
            k_values = [16]
            r_values = [0.05]
            l_values = [5e-05]
        elif dataname == "lastfm_user":
            k_values = [4]
            r_values = [0.01]
            l_values = [1e-05]
        elif dataname == "blog":
            k_values = [1]
            r_values = [0.01]
            l_values = [5e-05]
        elif dataname == "flickr":
            k_values = [1]
            r_values = [0.01]
            l_values = [1e-05]
        elif dataname == "epinion":
            k_values = [16]
            r_values = [0.05]
            l_values = [1e-05]

    
    if args.verbose == 1:
        csv_file = f'{result_dir}/run_ffm_{dataname}_k{k_values[0]}_r{r_values[0]}_l{l_values[0]}_validation.csv'
    elif args.verbose == 2:
        csv_file = f'{result_dir}/run_ffm_{dataname}_k{k_values[0]}_r{r_values[0]}_l{l_values[0]}_validation_auto_stop.csv'
    else:
        csv_file = f'{result_dir}/run_ffm_{dataname}_k{k_values[0]}_r{r_values[0]}_l{l_values[0]}.csv'

    mat_file = f'../../data/{dataname}/{dataname}.mat'
    data = scio.loadmat(mat_file)
    user_item = data['user_item'].tocoo().tocsr()
    num_users, num_items = user_item.shape

    dict_file = f'{actual_dir}/{dataname}_all_feature_dict.csv'

    for trial in trials:
        train_indices = data['train'][trial]
        test_indices = data['test'][trial]
        num_train_users = train_indices.shape[0]
        num_test_users = test_indices.shape[0]
        ffm_file = f'{actual_dir}/{dataname}_test_{trial}.ffm'
        actual = ffm_cal.parse_test_ffm_file(ffm_file, dict_file, num_test_users, num_items)
        # topks=[5, 10, 15, 20]
        topks=[10, 20]


        with open(csv_file, 'a+', newline='') as f_csv:
            writer = csv.writer(f_csv)
            p_lst = ['p' + str(i) for i in topks]
            r_lst = ['r' + str(i) for i in topks]
            n_lst = ['n' + str(i) for i in topks]
            h_lst = ['h' + str(i) for i in topks]
            header = ['trial'] + p_lst + r_lst + n_lst + h_lst
            if trial==0:
                writer.writerow(header)

            for k in k_values:
                for r in r_values:
                    for l in l_values:
                        predict_file = f'{predict_dir}/output_{dataname}_k{k}_r{r}_l{l}_{trial}.txt'
                        predict = ffm_cal.parse_output_file(predict_file, num_test_users, num_items)

                        precision_record, recall_record, ndcg_record, hitrate_record = ffm_cal.calculate_metrics(actual, predict, topks)                 
                        row = [trial] + precision_record + recall_record + ndcg_record + hitrate_record
                        writer.writerow(row)



if __name__ == "__main__":
    main()