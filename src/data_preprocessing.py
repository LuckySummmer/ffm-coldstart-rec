import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# only use user_item 只使用user_item
# import utils.ffm_format_data as ffm_utils

# use user_side and user_item
import utils.ffm_format_data2 as ffm_utils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser("data preprocessing: creating ffm_format_data")
    parser.add_argument('-d', '--dataname', type=str, default='delicious_user',
                        choices=['delicious_user', 'lastfm_user', 'flickr', 'epinion', 'blog'],
                        help='dataname to be handled')
    parser.add_argument('-i', '--input', type=str, default='../data/raw',
                        help='raw dataset data file dir')
    parser.add_argument('-o', '--output', type=str, default='../data/processed',
                        help='new created data file dir')
    parser.add_argument('-t', '--trials', action='store_true',
                        help='preprocess data of 10 trials')
    
    args = parser.parse_args()
    dataname = args.dataname
    input_dir = f'{args.input}/{dataname}'      
    output_dir = f'{args.output}/{dataname}'

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Make dir {output_dir} Done!")
    
    if args.trials:
        trials = list(range(10))
        print("RUN 10 trials.")
    else:
        trials = [0]
        print("TUNE the first trial.")
        
    
    mat_file = os.path.join(PROJECT_ROOT, 'data', 'raw', f'{dataname}', f'{dataname}.mat')
    output_feature_dict_file = os.path.join(PROJECT_ROOT, 'data', 'processed', f'{dataname}', f'{dataname}_all_feature_dict.csv')

    feature_dict = ffm_utils.construct_feature_dict(mat_file)
    ffm_utils.save_dict_to_csv(feature_dict, output_feature_dict_file)

    for trial in trials:

        output_train_file = os.path.join(PROJECT_ROOT, 'data', 'processed', f'{dataname}', f'{dataname}_train_{trial}.ffm')
        output_test_file = os.path.join(PROJECT_ROOT, 'data', 'processed', f'{dataname}', f'{dataname}_test_{trial}.ffm')

        output_train_feature_dict_file = os.path.join(PROJECT_ROOT, 'data', 'processed', f'{dataname}', f'{dataname}_train_feature_dict_{trial}.csv')
        output_test_feature_dict_file = os.path.join(PROJECT_ROOT, 'data', 'processed', f'{dataname}', f'{dataname}_test_feature_dict_{trial}.csv')

        train_feature_dict, test_feature_dict = ffm_utils.split_train_test_according_to_key(mat_file, feature_dict, trial)
        ffm_utils.save_dict_to_csv(train_feature_dict, output_train_feature_dict_file)
        ffm_utils.save_dict_to_csv(test_feature_dict, output_test_feature_dict_file)

        ffm_utils.get_ffm_dataset(
            mat_file,
            output_train_file,
            output_test_file,
            train_feature_dict,
            test_feature_dict,
            feature_dict,
            trial=trial
        )
        


if __name__ == "__main__":
    main()