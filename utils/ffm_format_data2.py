import pandas as pd
import scipy.io as scio


DICTFIELD = {}
DICTFIELD['USERID'] = 0
DICTFIELD['ITEMID'] = 1

def assign_feature_id(feature_dict, key, current_id):
    """
    为每个用户(field 1)和物品(field 2)分配唯一的feature_id
    """
    if key not in feature_dict:
        feature_dict[key] = current_id
        current_id += 1
    return current_id


def construct_feature_dict(mat_file):
    """
    构造特征字典, 返回DICTFEATURE
    """
    data = scio.loadmat(mat_file)

    user_item = data['user_item'].tocoo().tocsr()
    num_users, num_items = user_item.shape
    print(f"数据集包含了{num_users}个用户，{num_items}个物品")

    feature_dict = {}
    current_id = 0

    for user_id in range(num_users):
        field_0_key = f"USER{user_id}"
        current_id = assign_feature_id(feature_dict, field_0_key, current_id)
    
    for item_id in range(num_items):
        field_1_key = f"ITEM{item_id}"
        current_id = assign_feature_id(feature_dict, field_1_key, current_id)


    # use user_side data

    field_id = 2

    user_side = data['user_side'].tocoo().tocsr()
    _, num_user_feature = user_side.shape
    for feat_id in range(num_user_feature):
        field_key = f"FEATID{feat_id}"
        DICTFIELD[field_key] = field_id
        field_id = field_id + 1

        field_key2 = f"FEATID{feat_id}"
        current_id = assign_feature_id(feature_dict, field_key2, current_id)

            
    return feature_dict


def save_dict_to_csv(feature_dict, file_path):
    """
    保存字典为csv文件
    """
    df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['FeatureID'])
    df.to_csv(file_path, index_label='FeatureKey')
    print(f"特征字典已保存到{file_path}")


def load_dict_from_csv(file_path):
    """
    从csv加载特征字典
    """
    df = pd.read_csv(file_path, index_col='FeatureKey')
    return df['FeatureID'].to_dict()


def split_train_test_according_to_key(mat_file, all_feature_dict, trial=0):
    """
    拆分训练和测试集的特征字典，并保存到csv
    """

    data = scio.loadmat(mat_file)

    train_indices = data['train'][trial]
    test_indices = data['test'][trial]

    print("正在拆分训练和测试特征字典...")

    train_feature_dict = {
        f"USER{user_id}": all_feature_dict[f"USER{user_id}"] for user_id in train_indices
    }
    test_feature_dict = {
        f"USER{user_id}": all_feature_dict[f"USER{user_id}"] for user_id in test_indices
    }

    return train_feature_dict, test_feature_dict
        

def get_ffm_dataset(mat_file, output_train_file, output_test_file,
                    train_user_feature_dict, test_user_feature_dict, all_feature_dict, trial=0):
    """
    生成ffm数据集
    """
    data = scio.loadmat(mat_file)
    user_item = data['user_item'].tocoo().tocsr()
    user_side = data['user_side'].tocoo().tocsr()
    train_indices = data['train'][trial]
    test_indices = data['test'][trial]

    _, num_items = user_item.shape
    _, num_user_features = user_side.shape

    with open(output_train_file, 'w') as f_train, open(output_test_file, 'w') as f_test:
        for is_train, indices, output_file, user_feature_dict in [
            (True, train_indices, f_train, train_user_feature_dict),
            (False, test_indices, f_test, test_user_feature_dict)
        ]:
            for user_id in indices:
                for item_id in range(num_items):

                    label = int(user_item[user_id, item_id])

                    field_0_key = f"USER{user_id}"
                    field_1_key = f"ITEM{item_id}"

                    if field_0_key not in user_feature_dict or field_1_key not in all_feature_dict:
                        continue

                    feature = [
                        (0, user_feature_dict[field_0_key], 1),
                        (1, all_feature_dict[field_1_key], 1)
                    ]

                    # for user_side feature construction

                    start = user_side.indptr[user_id]
                    end = user_side.indptr[user_id + 1]
                    friend_id = list(user_side.indices[start:end])
                    for i in friend_id:
                        _field = DICTFIELD[f"FEATID{i}"]
                        _key = all_feature_dict[f"FEATID{i}"]


                        feature.append((_field, _key, 1))


                    feature_str = ' '.join(f"{i[0]}:{i[1]}:{i[2]}" for i in feature)
                    output_file.write(f"{label} {feature_str}\n")
    



# just for test
if __name__ == "__main__":
    mat_file = '../../data/delicious_user/delicious_user.mat'
    data = scio.loadmat(mat_file)
    trial = 0

    user_item = data['user_item'].tocoo().tocsr()
    user_side = data['user_side'].tocoo().tocsr()
    train_indices = data['train'][trial]
    test_indices = data['test'][trial]

    num_users, num_items = user_item.shape
    _, num_user_features = user_side.shape

    for user_id in range(num_users):
        # for item_id in range(num_items):

            # label = int(user_item[user_id, item_id])

            # field_0_key = f"USER{user_id}"
            # field_1_key = f"ITEM{item_id}"

            # if field_0_key not in user_feature_dict or field_1_key not in all_feature_dict:
            #     continue

            # feature = [
            #     (0, user_feature_dict[field_0_key], 1),
            #     (1, all_feature_dict[field_1_key], 1)
            # ]

            # for user_side feature construction

        start = user_side.indptr[user_id]
        end = user_side.indptr[user_id + 1]
        friend_id = list(user_side.indices[start:end])
        if user_id == 0:
            print(friend_id)


            # feature_str = ' '.join(f"{i[0]}:{i[1]}:{i[2]}" for i in feature)
            # output_file.write(f"{label} {feature_str}\n")

