import argparse
import os
import os.path as osp
import subprocess
from pathlib import Path


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

    parser = argparse.ArgumentParser("Run libffm using ffm format data")
    parser.add_argument('-d', '--dataname', type=str, default='delicious_user',
                        choices=['delicious_user', 'lastfm_user', 'flickr', 'epinion', 'blog',
                                 'delicious_user_1', 'lastfm_user_1',
                                 'delicious_user_2', 'lastfm_user_2'],
                        help='dataname to be run')
    parser.add_argument('-i', '--input', type=str, default='../dataset2',
                        help='Input directory containing ffm format data')
    parser.add_argument('-m', '--model', type=str, default='../model2',
                        help='Path to save models')
    parser.add_argument('-o', '--output', type=str, default='../output2',
                        help='Path to save outputs')
    parser.add_argument('-c', '--config', type=str, default='../params',
                        help='Path to hyperparameter configuration file')
    parser.add_argument('-t', '--trials', action='store_true',
                        help='run 10 trials')
    # 补充 加验证集的选项
    parser.add_argument('-v', '--validation', action='store_true',
                        help='Whether to use a validation set while training')
    # 补充 auto-stop 防止过拟合
    # blog 很有必要采用auto-stop，是因为跑得很慢呀，但不知道是否能有提升
    parser.add_argument('-a', '--auto_stop', action='store_true',
                        help='Whether to use auto-stop while training')

    args = parser.parse_args()
    dataname = args.dataname
    input_dir = f'{args.input}/{dataname}'
    model_dir = f'{args.model}/{dataname}'
    output_dir = f'{args.output}/{dataname}'

    if not osp.exists(model_dir):
        os.mkdir(model_dir)
        print(f"创建{model_dir}完成")

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        print(f"创建{output_dir}完成")


    
    if args.validation:
        print(f"训练过程中将采用验证集")
    
    if args.auto_stop:
        assert args.validation == True
        print(f"训练过程中将采用auto_stop来避免过拟合")


    config_path = f'{args.config}/params_{dataname}.txt'
    config = load_config(config_path)
    k_values = list(map(int, config['k_values'].split(',')))    # [4, 8, 12, 16]
    r_values = list(map(float, config['r_values'].split(',')))  # [0.01, 0.05, 0.1]
    l_values = list(map(float, config['l_values'].split(',')))  # [1e-05, 2e-05, 5e-05, 0.0001]
    data_sets = config['data_sets'].split(',')                  # ['delicious_user']
    assert data_sets[0] == dataname

    if args.trials:
        trials = list(range(1, 10))     # 从第二个trial开始跑，减少重复跑浪费时间
        print("Run 10 trials.")
        
        if dataname == "delicious_user":
            # 参数组合1 - 1
            # k_values = [16]
            # r_values = [0.1]
            # l_values = [5e-05]

            # 参数组合2 - 1
            # k_values = [16]
            # r_values = [0.05]
            # l_values = [1e-05]

            # 参数组合3 - 0
            k_values = [16]
            r_values = [0.05]
            l_values = [5e-05]
        elif dataname == 'epinion':
            # 参数组合1 - 1 - 623
            # k_values = [16]
            # r_values = [0.05]
            # l_values = [1e-05]

            # 参数组合2 - 1 - pazhou
            k_values = [8]
            r_values = [0.1]
            l_values = [1e-05]

        elif dataname == "blog":
            # 参数组合1 - 1 - 623
            k_values = [4]
            r_values = [0.01]
            l_values = [2e-05]

            # 参数组合2 - 1 - pazhou
            # k_values = [1]
            # r_values = [0.01]
            # l_values = [5e-05]


        elif dataname == "lastfm_user":
            # 参数组合1
            # k_values = [16]
            # r_values = [0.1]
            # l_values = [5e-05]

            # 参数组合2
            # k_values = [8]
            # r_values = [0.01]
            # l_values = [1e-05]

            # 参数组合3
            k_values = [4]
            r_values = [0.01]
            l_values = [1e-05]

        elif dataname == "flickr":
            # 参数组合1 - 1 - 623
            # k_values = [4]
            # r_values = [0.01]
            # l_values = [2e-05]

            # 参数组合2 - 0 - 623 and pazhou
            k_values = [1]
            r_values = [0.01]
            l_values = [1e-05]

            trials = list(range(1, 5))  # [1,2,3,4] 另外 pazhou [5,6,7,8,9]



        elif dataname == "lastfm_user_2":
            k_values = [12]
            r_values = [0.1]
            l_values = [5e-05]

    else:
        trials = [0]
        print("Tune the first trial.")


    for trial in trials:
        for k in k_values:
            for r in r_values:
                for l in l_values:
                    model_file = f'{model_dir}/model_{dataname}_k{k}_r{r}_l{l}_{trial}.ffm'

                    train_file = f'{input_dir}/{dataname}_train_{trial}.ffm'
                    test_file = f'{input_dir}/{dataname}_test_{trial}.ffm'
                    output_file = f'{output_dir}/output_{dataname}_k{k}_r{r}_l{l}_{trial}.txt'

                    print("------------------------training------------------------")

                    # train

                    if args.validation and args.auto_stop:
                        cmd_train = [
                            '../libffm/ffm-train',
                            '-k', str(k),
                            '-r', str(r),
                            '-l', str(l),
                            '--auto-stop',
                            '-p', str(test_file),
                            str(train_file),
                            str(model_file)
                        ]
                        print(f"Training {dataname} with k={k}, r={r}, l={l}, with validation set and auto-stop strategy.")
                    elif args.validation and not args.auto_stop:
                        cmd_train = [
                            '../libffm/ffm-train',
                            '-k', str(k),
                            '-r', str(r),
                            '-l', str(l),
                            '-p', str(test_file),
                            str(train_file),
                            str(model_file)
                        ]
                        print(f"Training {dataname} with k={k}, r={r}, l={l}, with validation set.")
                    else:
                        cmd_train = [
                            '../libffm/ffm-train',
                            '-k', str(k),
                            '-r', str(r),
                            '-l', str(l),
                            # '-t', str(10),        # for tuning epinion and blog less
                            # '-s', str(8),         # 提高运行效率 for flickr
                            # '-s', str(2),         # 提高运行效率 for lastfm_user run 10 trials 
                            # '-s', str(3),
                            # '-s', str(5),         # for blog and flickr run 10
                            # '-s', str(5),         # for time datasets: delicious_user_1 and lastfm_user_1
                            # '-s', str(5),         # for topN datasets: delciious_user_2 and lastfm_user_2
                            '-s', str(10),          # for tuning epinion and runing epinion
                            # '-s', str(8),         # for tuning and runing epinion
                            str(train_file),
                            str(model_file)
                        ]
                        print(f"Training {dataname} with k={k}, r={r}, l={l}")

                    print(f"Input data: {train_file}")
                    print(f"Model will be saved to: {model_file}")


                    try:
                        subprocess.run(cmd_train, check=True)
                        print("Training completed successfully")
                    except subprocess.CalledProcessError as e:
                        print(f"Training failed with error code {e.returncode}")
                    except FileNotFoundError:
                        print("Error: ffm-train executable not found")


                    print("------------------------predicting-----------------------")
                    print(f"Testing {dataname} with k={k}, r={r}, l={l}")
                    print(f"Input data: {test_file}")
                    print(f"Model will be loaded from: {model_file}")
                    print(f"Output will be saved to: {output_file}")
                    
                    # predict
                    cmd_test = [
                        '../libffm/ffm-predict',
                        str(test_file),
                        str(model_file),
                        str(output_file)
                    ]


                    try:
                        # 检查模型文件是否存在
                        if not osp.exists(model_file):
                            raise FileNotFoundError(f"Model file {model_file} not found")

                        subprocess.run(cmd_test, check=True)
                        print(f"Predictions saved to: {output_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Prediction failed with error code {e.returncode}")
                    except FileNotFoundError as e:
                        print(f"Error: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error: {str(e)}")

                    
                    print('--------------------------ending--------------------------\n\n')


if __name__ == "__main__":
    main()