# import ipdb
import os
import numpy as np
import json


data_dir = '/path_to_data/nwucla/'  # curie
lists_dir = '/path_to_lists_dir/'  # curie

dict_translate_actions = {1: 1, 2: 2, 3: 3,
                          4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 11: 9, 12: 10}


def get_tfrecords_nwucla(eval_mode, training_view_ids):
    test_list = []
    val_list = []
    train_list = []
    rgb_depth_dset_dir = 'multiview_action'

    recs = []
    for view in range(1, 4):
        videos = os.listdir(os.path.join(
            data_dir, rgb_depth_dset_dir, 'view_' + str(view)))
        videos = ['view_' + str(view) + '/' + x for x in videos]
        recs.extend(videos)

    for rec in recs:
        if int(rec[5:6]) in training_view_ids:
            train_list.append(rec)
        else:
            test_list.append(rec)

    # to get everytime the same sequence of training examples
    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)

    # balanced sampling validation set from training set (5%)
    n_classes = 10
    dict_actions = {}
    for i in range(n_classes):
        dict_actions[i] = []
    for video in train_list:
        print(video)
        dict_actions[dict_translate_actions[int(
            video[8:10])] - 1].append(video)

    val_list = []
    n_per_class = int(len(train_list) / n_classes * 0.05)
    for key, value in dict_actions.items():
        np.random.seed(key)
        choice_list = np.random.choice(value, n_per_class, replace=False)
        value = [x for x in value if x not in choice_list]
        dict_actions[key] = value
        val_list.extend(choice_list)

    train_list = []
    for key, value in dict_actions.items():
        train_list.extend(value)

    return train_list, val_list, test_list


# training_view_ids = [1, 2]
# training_view_ids = [1, 3]
training_view_ids = [2, 3]
train_list, val_list, test_list = get_tfrecords_nwucla(
    'x-view', training_view_ids)
view_dset = {'train': train_list, 'val': val_list, 'test': test_list}
dict_dset = {'view': view_dset}

fname = 'nwucla_train_2_3.txt'
filenames_dir = os.path.join(data_dir, 'filenames-lists')
if not os.path.exists(filenames_dir):
    os.makedirs(filenames_dir)
with open(os.path.join(filenames_dir, fname), 'w') as f:
    json.dump(dict_dset, f)

filenames_dir = os.path.join(lists_dir, 'codebase/preproc/lists/nwucla/')
if not os.path.exists(filenames_dir):
    os.makedirs(filenames_dir)
with open(os.path.join(filenames_dir, fname), 'w') as f:
    json.dump(dict_dset, f)

# ipdb.set_trace()
# with open(os.path.join(filenames_dir, fname), 'r') as f:
#     asd = json.load(f)
#
# ipdb.set_trace()
print('finished')
