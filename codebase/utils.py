import tensorflow as tf
import os
import time
import numpy as np
import argparse
import json
import sys

# the data input is organized as follows:
# Previously to training, one time only:
# (1) It is generated a txt list of training, validation, and testing samples. This list is generated once. It refers to a particular setup, e.g. training view 1 and 2, test on view 3.
# (2) It is generated one tfrecord per video, which contains the 3 arrays,
# each of these containing the path for each frame of RGB, depth, and flow. These are used for training.
# (3) It is generated one tfrecord per video containing 10 arrays, each of this containing the path to 8 frames. These are used for testing and validation.
# data modality.
# For training:
# At the beginning of each experiment, the script reads this list, and build 3 arrays with the paths to the tfrecords for training, validation, and testing.
# The parser script reads the tfrecords and build the batch of samples.


LOG_PATH = ''
CKPT_PATH = ''


def get_train_test_path(dset, notes=''):
    if dset == 'nwucla':
        train_data_path = 'path/train_tfrecords/'
        test_data_path = '/path/testval_tfrecords/'
        list_path = '/path_to_list/nwucla_train_2_3.txt'
    if 'uwa3dii' in dset:
        train_data_path = 'path/train_tfrecords/'
        test_data_path = '/path/testval_tfrecords/'
        list_path = '/path_to_list/uwa3dii_train_.txt'
    elif dset == 'ntu120':
        train_data_path = 'path/train_tfrecords/'
        test_data_path = '/path/testval_tfrecords/'
        list_path = '/path_to_list/'
    elif dset == 'ntu':
        train_data_path = 'path/train_tfrecords/'
        test_data_path = '/path/testval_tfrecords/'
        list_path = '/path_to_list/'

    return train_data_path, test_data_path, list_path


def get_nwucla_fnames(mini=False):
    # gets the path to the tfrecords, according to the machine that is running
    # on
    train_data_path, test_data_path, list_path = get_train_test_path(
        dset='nwucla')

    # reads what samples belong to training set, val set, and test set.
    with open(list_path, 'r') as f:
        list_content = json.load(f)

    train_fnames = list_content['view']['train']
    tmp_val_fnames = list_content['view']['val']
    tmp_test_fnames = list_content['view']['test']

    # builds the full path to the tfrecords
    train_fnames = list(map(lambda x: os.path.join(
        train_data_path, x + '.tfrecord'), train_fnames))

    tmp_val_fnames = list(map(lambda x: os.path.join(
        test_data_path, x), tmp_val_fnames))
    tmp_test_fnames = list(map(lambda x: os.path.join(
        test_data_path, x), tmp_test_fnames))

    val_fnames = []
    test_fnames = []
    for i in range(1, 11):
        val_fnames.extend(list(map(lambda x: x + '_clip'
                                   + str(i).zfill(2) + '.tfrecord', tmp_val_fnames)))
        test_fnames.extend(list(map(lambda x: x + '_clip'
                                    + str(i).zfill(2) + '.tfrecord', tmp_test_fnames)))

    train_fnames.sort()
    np.random.seed(0)
    np.random.shuffle(train_fnames)

    train_fnames = [x for x in train_fnames if os.path.exists(x)]
    val_fnames = [x for x in val_fnames if os.path.exists(x)]
    test_fnames = [x for x in test_fnames if os.path.exists(x)]

    return train_fnames, val_fnames, test_fnames


def get_uwa3d_filenames(mini=False, dset='', notes=''):
    train_data_path, test_data_path, list_path = get_train_test_path(
        dset=dset, notes=notes)
    with open(list_path, 'r') as f:
        dset_list = json.load(f)

    train_list = dset_list['view']['train']
    val_list = dset_list['view']['val']
    test_list = dset_list['view']['test']

    train_list = list(map(lambda x: os.path.join(
        train_data_path, 'a' + x[1:3], x + '.tfrecord'), train_list))

    new_test_list = []
    for video in test_list:
        for i in range(10):
            new_test_list.append(os.path.join(
                test_data_path, 'a' + video[1:3], video + '_clip' + str(i + 1).zfill(2) + '.tfrecord'))

    new_val_list = []
    for video in val_list:
        for i in range(10):
            new_val_list.append(os.path.join(
                test_data_path, 'a' + video[1:3], video + '_clip' + str(i + 1).zfill(2) + '.tfrecord'))

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)

    train_list = [x for x in train_list if os.path.exists(x)]
    new_val_list = [x for x in new_val_list if os.path.exists(x)]
    new_test_list = [x for x in new_test_list if os.path.exists(x)]

    return train_list, new_val_list, new_test_list


def get_dset_filenames_step2(dset='ntu', argsmini=False, notes=''):
    # get all modalities
    if 'uwa3d' in dset:
        train_fnames, val_fnames, test_fnames = get_uwa3d_filenames(
            mini=argsmini, dset=dset, notes=notes)
    elif 'ntu' == dset:
        train_fnames, val_fnames, test_fnames = get_ntu60_from120_fnames(
            'cross_subj', mini=argsmini)
    elif 'nwucla' in dset:
        train_fnames, val_fnames, test_fnames = get_nwucla_fnames(
            mini=argsmini)
    elif 'ntu120' == dset:
        train_fnames, val_fnames, test_fnames = get_ntu120_fnames(
            'cross_subj', mini=argsmini)
    return train_fnames, val_fnames, test_fnames


def get_ntu_filenames_s2(dset, mini=False):
    train_data_path, ntu_testval_dir, list_path = get_train_test_path(
        dset='ntu')
    ntu_xmini_list = 'path_to/lists/fivepc_ntu_list.txt'
    full_ntu_list = 'path_to/lists/full_ntu_list.txt'
    ntu_xmini_list = 'path_to/lists/tenpc_ntu_list.txt'

    if dset == 'ntu':
        ntu_list = full_ntu_list
    elif dset == 'ntu-xmini':
        ntu_list = ntu_xmini_list
    with open(ntu_list, 'r') as f:
        dset_list = json.load(f)

    ntu_dir = '/tfrecords_names/'
    train_list = dset_list['subj']['train']
    test_list = dset_list['subj']['test']
    val_list = dset_list['subj']['val']

    train_list = list(map(lambda x: os.path.join(
        ntu_dir, 'action_' + x[-3:], x + '.tfrecord'), train_list))

    new_val_list = []
    for video in val_list:
        for i in range(10):
            new_val_list.append(os.path.join(
                ntu_testval_dir, 'action_' + video[-3:], video + '_clip' + str(i + 1).zfill(3) + '.tfrecord'))
    new_test_list = []
    for video in test_list:
        for i in range(10):
            new_test_list.append(os.path.join(
                ntu_testval_dir, 'action_' + video[-3:], video + '_clip' + str(i + 1).zfill(3) + '.tfrecord'))

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)
    return train_list, new_val_list, new_test_list


def get_ntu60_from120_fnames(eval_mode, mini):
    train_list, val_fnames, test_fnames = get_ntu120_fnames(eval_mode, mini)
    train_list = [x for x in train_list if int(x[-12:-9]) < 61]
    val_fnames = [x for x in val_fnames if int(x[-20:-17]) < 61]
    test_fnames = [x for x in test_fnames if int(x[-20:-17]) < 61]
    return train_list, val_fnames, test_fnames


def get_ntu120_fnames(eval_mode, mini):
    train_data_path, test_data_path, list_path = get_train_test_path(
        dset='ntu120')
    list_name = 'all_ntu120_list.txt'

    with open(os.path.join(list_path, list_name), 'r') as f:
        dset_list = json.load(f)

        train_list = dset_list['subj']['train']
        # len = 60240
        tmp_val_fnames = dset_list['subj']['val']
        tmp_test_fnames = dset_list['subj']['test']

    if mini:
        # 50% of the data
        n_per_class = int(len(train_list) / 2 / 120)
        dictt = {}
        for i in range(1, 121):
            dictt[i] = []

        for x in train_list:
            classs = int(x[-3:])
            dictt[classs].append(x)

        new_train = []
        for key, value in dictt.items():
            np.random.seed(key)
            choice_list = np.random.choice(value, n_per_class, replace=False)
            new_train.extend(choice_list)
        train_list = new_train

    train_list = list(map(lambda x: os.path.join(
        train_data_path, 'action_' + x[-3:], x + '.tfrecord'), train_list))
    tmp_val_fnames = list(map(lambda x: os.path.join(
        test_data_path, 'action_' + x[-3:], x), tmp_val_fnames))
    tmp_test_fnames = list(map(lambda x: os.path.join(
        test_data_path, 'action_' + x[-3:], x), tmp_test_fnames))

    val_fnames = []
    test_fnames = []
    for i in range(1, 11):
        val_fnames.extend(list(map(lambda x: x + '_clip'
                                   + str(i).zfill(3) + '.tfrecord', tmp_val_fnames)))
        test_fnames.extend(list(map(lambda x: x + '_clip'
                                    + str(i).zfill(3) + '.tfrecord', tmp_test_fnames)))

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)
    return train_list, val_fnames, test_fnames


######################################################################
######################################################################
def get_arguments():
    parser = argparse.ArgumentParser(description='args...')
    # --dset=[nwucla/uwa3dii/ntu/ntu-mini]
    parser.add_argument('--dset', action='store',
                        dest='dset', default='ntu-xmini')
    #  --eval=[cross_subj/cross_view]
    parser.add_argument('--eval', action='store',
                        dest='eval_mode', default='cross_subj')
    # batch size = number of videos. We sample 5 frames from each video.
    parser.add_argument('--batch_sz', action='store',
                        dest='batch_sz', default='8', type=int)
    parser.add_argument('--n_epochs', action='store',
                        dest='n_epochs', default='220', type=int)
    parser.add_argument('--lr', action='store',
                        dest='learning_rate', default='0.001', type=float)
    parser.add_argument('--ckpt', action='store',
                        dest='ckpt', default='')
    parser.add_argument('--temp', action='store',
                        dest='temp', default='2', type=int)
    parser.add_argument('--a_distill_l', action='store',
                        dest='a_distill_l', default='1.0', type=float)
    parser.add_argument('--a_loser_gt', action='store',
                        dest='a_loser_gt', default='1.0', type=float)
    parser.add_argument('--gpu0', action='store',
                        dest='gpu0', default='/gpu:0')
    parser.add_argument('--optimizer', action='store',
                        dest='optimizer', default='Momentum')
    parser.add_argument('--notes', action='store',
                        dest='notes', default='')
    parser.add_argument('--n_frames', action='store',
                        dest='n_frames', default='8', type=int)
    parser.add_argument('--step_summ', action='store',
                        dest='step_summ', default='500', type=int)
    parser.add_argument('--dry', action='store_true',
                        dest='dryrun', default=False)
    parser.add_argument('--mini', action='store_true',
                        dest='argsmini', default=False)
    args = parser.parse_args()
    return args


def get_experiment_id(prefix, **kwargs):
    datetime = time.strftime("%d%m%Y_%H%M%S")
    hyperparameters = ''.join('_' + str(value)
                              for _, value in sorted(kwargs.items()))
    experiment_id = prefix + '_' + datetime + '_' + hyperparameters
    return experiment_id


def get_log_ckpt_path(dryrun):
    log_path = LOG_PATH
    ckpt_path = CKPT_PATH
    return log_path, ckpt_path


def create_folders(prefix, dry_run, **kwargs):
    experiment_id = get_experiment_id(prefix, **kwargs)
    log_path, ckpt_path = get_log_ckpt_path(dry_run)
    os.makedirs(os.path.join(ckpt_path, kwargs['dset'], experiment_id))
    os.makedirs(os.path.join(log_path, kwargs['dset'], experiment_id))
    return experiment_id
######################################################################
######################################################################


def get_n_classes(dset):
    n_classes_dict = {'ntu120': 120,
                      'ntu': 60,
                      'ntu-mini': 60,
                      'ntu-xmini': 60,
                      'uwa3dii': 30,
                      'nwucla': 10
                      }
    return n_classes_dict[dset]


def get_nwucla_class(org_class):
    # because the dataset filenames class numbers are not from 1 to 10.
    dict_translate_actions = {1: 1, 2: 2, 3: 3,
                              4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 11: 9, 12: 10}
    return dict_translate_actions[org_class]
######################################################################
######################################################################


def init_f_log(log_path, exp_id, sys_argv):
    f_log = open(os.path.join(log_path, 'log.txt'), 'a')
    double_log(f_log, '\n###############################################\n' +
               exp_id + '\n#####################################\n')
    f_log.write(' '.join(sys_argv[:]) + '\n')
    f_log.flush()
    return f_log


def double_log(f_log, string):
    datetime = time.strftime("%d%m%Y_%H%M%S")
    string = datetime + ' ' + string
    print(string.rstrip())
    f_log.write(string)
    f_log.flush()
######################################################################
######################################################################


def correct_pred(net, y):
    return tf.equal(
        tf.argmax(net, 1), tf.argmax(y, 1))


def accuracy(net, y):
    return tf.reduce_mean(
        tf.cast(correct_pred(net, y), tf.float32))


def learning_rate_with_decay(batch_size, batch_denom, num_images, boundary_epochs, decay_rates, base_lr=0.1, warmup=False):
    # from tf resnet official github
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.
      base_lr: Initial learning rate scaled based on batch_denom.
      warmup: Run a 5 epoch warmup to the initial lr.
    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.train.piecewise_constant(
            global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
                    warmup_steps, tf.float32))
            return tf.cond(pred=global_step < warmup_steps,
                           true_fn=lambda: warmup_lr,
                           false_fn=lambda: lr)
        return lr  # , initial_learning_rate, batches_per_epoch, vals

    return learning_rate_fn
