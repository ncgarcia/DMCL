import os
import tensorflow as tf
from codebase import utils
from codebase import parsers
from codebase import restorers
import numpy as np
import sys

# tensorflow models_dir ##################################################
sys.path.insert(0, './nets/')
args = utils.get_arguments()
import resnet_official.resnet_3D as resnet
###########################################################################


def train(exp_id, train_fnames, val_fnames, test_fnames, n_classes):
    log_path, ckpt_path = utils.get_log_ckpt_path(args.dryrun)
    log_path = os.path.join(log_path, args.dset, exp_id)
    ckpt_path = os.path.join(ckpt_path, args.dset, exp_id)

    # dataset ######################################################
    with tf.device('/cpu:0'):
        dset_train = tf.data.TFRecordDataset(
            train_fnames, compression_type="GZIP")
        seed = tf.placeholder(tf.int64, shape=())  # =epoch
        dset_train = dset_train.shuffle(100, seed=seed)
        dset_train = dset_train.map(
            lambda x: parsers._parse_mult_frame_s2(x, rescale=False), num_parallel_calls=8)
        dset_train = dset_train.batch(args.batch_sz, drop_remainder=True)
        dset_train = dset_train.prefetch(buffer_size=10)

        dset_val = tf.data.TFRecordDataset(
            val_fnames, compression_type="GZIP")
        dset_val = dset_val.map(
            lambda x: parsers._parse_mult_frame_test_allmods(x, rescale=False), num_parallel_calls=8)
        dset_val = dset_val.batch(args.batch_sz)
        dset_val = dset_val.prefetch(buffer_size=10)

        dset_test = tf.data.TFRecordDataset(
            test_fnames, compression_type="GZIP")
        dset_test = dset_test.map(
            lambda x: parsers._parse_mult_frame_test_allmods(x, rescale=False), num_parallel_calls=8)
        dset_test = dset_test.batch(args.batch_sz)
        dset_test = dset_test.prefetch(buffer_size=10)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       dset_train.output_types, dset_train.output_shapes)

        train_it = dset_train.make_initializable_iterator()
        val_it = dset_val.make_initializable_iterator()
        test_it = dset_test.make_initializable_iterator()
        next_element = iterator.get_next()
        batch_rgb = next_element[0]
        batch_depth = next_element[1]
        batch_oflow = next_element[2]
        batch_labels_raw = next_element[3]  # video labels
        batch_labels = tf.one_hot(batch_labels_raw, n_classes)
        batch_video_id = next_element[4]

    # tf Graph input ##############################################
    is_training = tf.placeholder(tf.bool, name="is_training")
    # with tf.device('/device:GPU:0'):
    net_oflow = resnet.Model(resnet_size=18,
                             bottleneck=False,  # resnet original bottleneck, not ours
                             num_classes=n_classes,
                             num_filters=64,
                             kernel_size=7,
                             conv_stride=2,
                             first_pool_size=3,
                             first_pool_stride=2,
                             block_sizes=[2, 2, 2, 2],
                             block_strides=[1, 2, 2, 2],
                             temporal_strides=[1, 2, 2, 2],
                             resnet_version=1,
                             data_format='channels_last',
                             dtype=tf.float32,
                             n_frames=args.n_frames,
                             name='resnet_oflow')

    # with tf.device('/device:GPU:1'):
    net_rgb = resnet.Model(resnet_size=18,
                           bottleneck=False,  # resnet original bottleneck, not ours
                           num_classes=n_classes,
                           num_filters=64,
                           kernel_size=7,
                           conv_stride=2,
                           first_pool_size=3,
                           first_pool_stride=2,
                           block_sizes=[2, 2, 2, 2],
                           block_strides=[1, 2, 2, 2],
                           temporal_strides=[1, 2, 2, 2],
                           resnet_version=1,
                           data_format='channels_last',
                           dtype=tf.float32,
                           n_frames=args.n_frames,
                           name='resnet_rgb')

    # with tf.device('/device:GPU:2'):
    net_depth = resnet.Model(resnet_size=18,
                             bottleneck=False,  # resnet original bottleneck, not ours
                             num_classes=n_classes,
                             num_filters=64,
                             kernel_size=7,
                             conv_stride=2,
                             first_pool_size=3,
                             first_pool_stride=2,
                             block_sizes=[2, 2, 2, 2],
                             block_strides=[1, 2, 2, 2],
                             temporal_strides=[1, 2, 2, 2],
                             resnet_version=1,
                             data_format='channels_last',
                             dtype=tf.float32,
                             n_frames=args.n_frames,
                             name='resnet_depth')

    logits_oflow, reps_oflow = net_oflow(
        batch_oflow, training=is_training, output_rep=True)
    logits_depth, reps_depth = net_depth(
        batch_depth, training=is_training, output_rep=True)
    logits_rgb, reps_rgb = net_rgb(
        batch_rgb, training=is_training, output_rep=True)
    logits_fused = logits_rgb + logits_depth + logits_oflow

    ############################################################
    ############################################################
    # from https://github.com/chhwang/cmcl/blob/master/src/model.py
    logits_list = [logits_rgb, logits_depth, logits_oflow]
    closs_list = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels_raw)
                  for logits in logits_list]
    # closs_list is a list with three elements, each of which is shape (batch_sz,)
    # min_index.shape=(batch_sz,) : indicates the stream with the min loss
    _, min_index = tf.nn.top_k(-tf.transpose(closs_list), 1)
    min_index = tf.transpose(min_index)

    ############################################################
    # SOFT
    # weight of loser distillation loss
    placeh_a_distill_l = tf.placeholder(tf.float32)
    # weight of loser ground truth loss
    placeh_a_loser_gt = tf.placeholder(tf.float32)
    placeh_temp = tf.placeholder(tf.float32)

    soft_softmax = [tf.nn.softmax(logits / placeh_temp)
                    for logits in logits_list]
    soft_winners = []
    for i in range(args.batch_sz):
        soft_winners.append(tf.gather_nd(
            soft_softmax, [min_index[0][i], i]))

    soft_loss_win = 0
    soft_loss_loser = 0
    for m in range(3):
        total_condition = tf.constant(
            [False] * args.batch_sz, dtype=tf.bool)
        topk = 0
        # true if this modality m is the winner
        condition = tf.equal(min_index[topk], m)
        total_condition = tf.logical_or(
            total_condition, condition)  # used for when topWinners> 0
        new_labels2 = tf.where(
            condition, batch_labels, soft_winners)  # true, false
        new_logits = tf.where(
            condition, logits_list[m], logits_list[m] / placeh_temp)

        loss_win = \
            tf.where(total_condition,
                     tf.stack([1.] * args.batch_sz),
                     tf.stack([placeh_a_loser_gt] * args.batch_sz)) * \
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(batch_labels), logits=logits_list[m])
        soft_loss_win += tf.reduce_mean(loss_win)
        # placeh_a_loser_gt is the weitght for ground_truth labels for losers.

        loss_losers = \
            tf.where(total_condition,
                     tf.stack([0.] * args.batch_sz),
                     tf.stack([placeh_a_distill_l] * args.batch_sz)) * \
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(new_labels2), logits=new_logits)
        soft_loss_loser += tf.reduce_mean(loss_losers)
        # placeh_a_distill_l is the distillation weight to the losers

    # scaling loss due to distillation
    soft_loss_loser = soft_loss_loser * placeh_temp * placeh_temp
    loss = soft_loss_loser + soft_loss_win
    ###########################################################

    def exclude_batch_norm(name):
        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        return 'batch_normalization' not in name and 'bias' not in name
    loss_filter_fn = exclude_batch_norm

    weight_decay = 1e-4
    l2_loss_of = weight_decay * tf.add_n([
        tf.nn.l2_loss(tf.cast(v, tf.float32))
        for v in tf.trainable_variables()
        if loss_filter_fn(v.name) and
        'flow' in v.name
    ])
    l2_loss_depth = weight_decay * tf.add_n([
        tf.nn.l2_loss(tf.cast(v, tf.float32))
        for v in tf.trainable_variables()
        if loss_filter_fn(v.name) and
        'depth' in v.name
    ])
    l2_loss_rgb = weight_decay * tf.add_n([
        tf.nn.l2_loss(tf.cast(v, tf.float32))
        for v in tf.trainable_variables()
        if loss_filter_fn(v.name) and
        'rgb' in v.name
    ])

    loss += l2_loss_of + l2_loss_rgb + l2_loss_depth

    global_step = tf.train.get_or_create_global_step()
    if args.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=args.learning_rate)
        learning_rate = args.learning_rate

    if args.optimizer == 'Momentum':
        # base_lr default is .0128
        lr_fn = utils.learning_rate_with_decay(
            batch_size=args.batch_sz, batch_denom=args.batch_sz,
            num_images=len(train_fnames), boundary_epochs=[100, 150, 180, 200],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=.00128)
        learning_rate = lr_fn(global_step)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9
        )
    if args.optimizer == 'Momentum-finetune':
        # base_lr default is .0128
        lr_fn = utils.learning_rate_with_decay(
            batch_size=args.batch_sz, batch_denom=args.batch_sz,
            num_images=len(train_fnames), boundary_epochs=[100, 150, 180, 200],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=.0000128)
        learning_rate = lr_fn(global_step)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9
        )

    grad_vars = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.get_collection(
        tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    acc_train = utils.accuracy(logits_fused, batch_labels)
    acc_train_rgb = utils.accuracy(logits_rgb, batch_labels)
    acc_train_depth = utils.accuracy(logits_depth, batch_labels)
    acc_train_flow = utils.accuracy(logits_oflow, batch_labels)

    ########### SUMMARIES ######################################
    for gradient, variable in grad_vars:
        if '/dense/' in variable.name or 'conv3d_1/' in variable.name:
            tf.summary.scalar(
                "norm/gradients/" + variable.name, tf.norm(gradient))
            tf.summary.scalar(
                "norm/variables/" + variable.name, tf.norm(variable))
            tf.summary.histogram("values/variables/"
                                 + variable.name, tf.reshape(variable, [-1]))
            tf.summary.histogram("values/gradients/"
                                 + variable.name, tf.reshape(gradient, [-1]))

    # reduce_mean to get the mean for the batch
    tf.summary.scalar('norm/logits_rgb',
                      tf.reduce_mean(tf.map_fn(tf.norm, logits_rgb)))
    tf.summary.scalar('norm/logits_depth',
                      tf.reduce_mean(tf.map_fn(tf.norm, logits_depth)))
    tf.summary.scalar('norm/logits_of',
                      tf.reduce_mean(tf.map_fn(tf.norm, logits_oflow)))
    summaries_norms_logits = tf.summary.merge_all(scope='norm/')
    tf.summary.histogram('values/logits_rgb',
                         tf.reshape(logits_rgb, [-1]))
    tf.summary.histogram('values/logits_depth',
                         tf.reshape(logits_depth, [-1]))
    tf.summary.histogram('values/logits_of',
                         tf.reshape(logits_oflow, [-1]))
    summaries_vars = tf.summary.merge_all(scope='values/')
    # tf.summary.scalar('loss/xentropy', loss_total)
    # tf.summary.scalar('loss/xentropy_rgb', cross_entropy_rgb)
    # tf.summary.scalar('loss/xentropy_depth', cross_entropy_depth)
    # tf.summary.scalar('loss/xentropy_of', cross_entropy_of)
    # tf.summary.scalar('loss/l2', l2_loss)
    tf.summary.scalar('loss/total', loss)
    summaries_losses = tf.summary.merge_all(scope='loss/')
    tf.summary.scalar('acc/train', acc_train)
    tf.summary.scalar('acc/train_rgb', acc_train_rgb)
    tf.summary.scalar('acc/train_depth', acc_train_depth)
    tf.summary.scalar('acc/train_flow', acc_train_flow)
    summaries_acc_train = tf.summary.merge_all(scope='acc/')
    summ_lr = tf.summary.scalar('learning_rate', learning_rate)
    summ_train = tf.summary.merge(
        [summaries_norms_logits, summaries_vars, summ_lr, summaries_acc_train, summaries_losses])
    accuracy_value_ = tf.placeholder(tf.float32, shape=())
    summ_acc_test_oracle = tf.summary.scalar(
        'acc_test_oracle', accuracy_value_)
    summ_acc_test_sum = tf.summary.scalar('acc_test_sum', accuracy_value_)
    summ_acc_test_rgb = tf.summary.scalar('acc_test_rgb', accuracy_value_)
    summ_acc_test_flow = tf.summary.scalar('acc_test_flow', accuracy_value_)
    summ_acc_test_depth = tf.summary.scalar('acc_test_depth', accuracy_value_)
    summ_acc_val_oracle = tf.summary.scalar('acc_val_oracle', accuracy_value_)
    summ_acc_val_sum = tf.summary.scalar('acc_val_sum', accuracy_value_)
    summ_acc_val_rgb = tf.summary.scalar('acc_val_rgb', accuracy_value_)
    summ_acc_val_flow = tf.summary.scalar('acc_val_flow', accuracy_value_)
    summ_acc_val_depth = tf.summary.scalar('acc_val_depth', accuracy_value_)
#################################################

    test_saver = tf.train.Saver(max_to_keep=3)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    with tf.Session(config=tf_config) as sess:
        train_handle = sess.run(train_it.string_handle())
        val_handle = sess.run(val_it.string_handle())
        test_handle = sess.run(test_it.string_handle())
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        f_log = utils.init_f_log(log_path, exp_id, sys.argv[:])

        def val_test(value_step, mode='val'):
            if mode == 'val' and val_fnames:
                utils.double_log(f_log, "eval val set \n")
                sess.run(val_it.initializer)
                step_handle = val_handle
                step_samples = len(val_fnames) / 10
                step_summ_rgb = summ_acc_val_rgb
                step_summ_depth = summ_acc_val_depth
                step_summ_flow = summ_acc_val_flow
                step_summ_sum = summ_acc_val_sum
                step_summ_oracle = summ_acc_val_oracle
            elif mode == 'test':
                utils.double_log(f_log, "eval test set \n")
                sess.run(test_it.initializer)
                step_handle = test_handle
                step_samples = len(test_fnames) / 10
                step_summ_rgb = summ_acc_test_rgb
                step_summ_depth = summ_acc_test_depth
                step_summ_flow = summ_acc_test_flow
                step_summ_sum = summ_acc_test_sum
                step_summ_oracle = summ_acc_test_oracle
            else:
                return -1
            try:
                dict_sum = {}
                dict_rgb = {}
                dict_of = {}
                dict_depth = {}
                hist_acc_per_class = np.zeros([5, n_classes])
                n_samples_per_class = np.zeros(n_classes)
                while True:
                    logits_sum, logits_r, logits_d, logits_f, video_id_val = sess.run([logits_fused, logits_rgb, logits_depth, logits_oflow, batch_video_id], feed_dict={
                        handle: step_handle, is_training: False})
                    for i in range(len(video_id_val)):
                        v = video_id_val[i]
                        if v in dict_sum:
                            dict_sum[v] = dict_sum[v] + logits_sum[i]
                        else:
                            dict_sum[v] = logits_sum[i]
                            if 'ntu' in args.dset:
                                this_class = int(v[-3:]) - 1
                            elif 'nwucla' in args.dset:
                                this_class = utils.get_nwucla_class(
                                    int(v[8:10])) - 1
                            elif 'uwa3dii' in args.dset:
                                this_class = int(v[1:3]) - 1
                            n_samples_per_class[this_class] += 1
                        if v in dict_rgb:
                            dict_rgb[v] = dict_rgb[v] + logits_r[i]
                        else:
                            dict_rgb[v] = logits_r[i]
                        if v in dict_of:
                            dict_of[v] = dict_of[v] + logits_f[i]
                        else:
                            dict_of[v] = logits_f[i]
                        if v in dict_depth:
                            dict_depth[v] = dict_depth[v] + logits_d[i]
                        else:
                            dict_depth[v] = logits_d[i]
            except tf.errors.OutOfRangeError:
                accum_correct_sum = 0
                accum_correct_r = 0
                accum_correct_d = 0
                accum_correct_f = 0
                accum_correct_oracle = 0
                for key in dict_sum:
                    dict_sum[key] = dict_sum[key] / 10
                    dict_rgb[key] = dict_rgb[key] / 10
                    dict_depth[key] = dict_depth[key] / 10
                    dict_of[key] = dict_of[key] / 10
                    pred_sum = np.argmax(dict_sum[key])
                    pred_r = np.argmax(dict_rgb[key])
                    pred_d = np.argmax(dict_depth[key])
                    pred_f = np.argmax(dict_of[key])
                    if 'ntu' in args.dset:
                        lab = int(key[-3:]) - 1
                    elif 'uwa3d' in args.dset:
                        lab = int(key[1:3]) - 1
                    elif 'nwucla' in args.dset:
                        lab = utils.get_nwucla_class(int(key[8:10])) - 1
                    if lab == pred_sum:
                        accum_correct_sum += 1
                        hist_acc_per_class[0, lab] += 1
                    if lab == pred_r:
                        hist_acc_per_class[2, lab] += 1
                        accum_correct_r += 1
                    if lab == pred_d:
                        hist_acc_per_class[3, lab] += 1
                        accum_correct_d += 1
                    if lab == pred_f:
                        hist_acc_per_class[4, lab] += 1
                        accum_correct_f += 1
                    if lab in [pred_f, pred_r, pred_d]:
                        accum_correct_oracle += 1
                        hist_acc_per_class[1, lab] += 1

                for i in range(n_classes):
                    value = n_samples_per_class[i]
                    hist_acc_per_class[:, i] = hist_acc_per_class[:, i] / value

                utils.double_log(
                    f_log, 'accuracy per class: 0sum 1oracle 2rgb 3depth 4flow \n')
                utils.double_log(f_log, '\n' + np.array2string(
                    hist_acc_per_class) + '\n')

                step_acc = accum_correct_r / step_samples
                utils.double_log(f_log, 'rgb Accuracy = %s \n' % str(step_acc))
                summary_acc = sess.run(step_summ_rgb, feed_dict={
                    accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)

                step_acc = accum_correct_d / step_samples
                utils.double_log(
                    f_log, 'depth Accuracy = %s \n' % str(step_acc))
                summary_acc = sess.run(step_summ_depth, feed_dict={
                    accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)

                step_acc = accum_correct_f / step_samples
                utils.double_log(f_log, 'flow Accuracy = %s \n' %
                                 str(step_acc))
                summary_acc = sess.run(step_summ_flow, feed_dict={
                    accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)

                step_acc = accum_correct_sum / step_samples
                summary_acc = sess.run(step_summ_sum, feed_dict={
                    accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)
                utils.double_log(f_log, 'sum Accuracy = %s \n' % str(step_acc))

                step_acc = accum_correct_oracle / step_samples
                summary_acc = sess.run(step_summ_oracle, feed_dict={
                    accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)
                utils.double_log(
                    f_log, 'oracle Accuracy = %s \n' % str(step_acc))
            return step_acc

        period_val_acc = 5  # periodicity for validation acc
        patience = 5  # how many (period_val_acc * patience) epochs til give up
        validation_accuracies = [0] * patience
        n_step = 0
        best_step = 0

        for epoch in range(args.n_epochs):
            utils.double_log(f_log, 'epoch %s \n' % str(epoch))
            sess.run(train_it.initializer, feed_dict={seed: epoch})
            accum_min_index = np.zeros(3)
            hist_classes = np.zeros([3, n_classes])
            a_distill_l = args.a_distill_l
            a_loser_gt = args.a_loser_gt
            temp = args.temp
            try:
                while True:
                    if n_step % args.step_summ == 0:  # get summaries
                        batch_labels_raw_val, min_index_val, _, summary = sess.run(
                            [batch_labels_raw, min_index, train_op, summ_train], feed_dict={handle: train_handle, is_training: True, placeh_a_distill_l: a_distill_l, placeh_a_loser_gt: a_loser_gt, placeh_temp: temp})
                        min_index_val = np.squeeze(min_index_val)
                        min_index_val = [int(x) for x in min_index_val]
                        for i, j in enumerate(min_index_val):
                            accum_min_index[j] += 1
                            hist_classes[j][batch_labels_raw_val[i]] += 1
                        summary_writer.add_summary(summary, n_step)
                    else:
                        batch_labels_raw_val, min_index_val, _ = sess.run([batch_labels_raw, min_index, train_op], feed_dict={
                            handle: train_handle, is_training: True, placeh_a_distill_l: a_distill_l, placeh_a_loser_gt: a_loser_gt, placeh_temp: temp})
                        min_index_val = np.squeeze(min_index_val)
                        min_index_val = [int(x) for x in min_index_val]
                        for i, j in enumerate(min_index_val):
                            accum_min_index[j] += 1
                            hist_classes[j][batch_labels_raw_val[i]] += 1
                    n_step = n_step + 1
            except tf.errors.OutOfRangeError:
                utils.double_log(f_log, 'nets: rgb depth oflow \n')
                utils.double_log(
                    f_log, 'total number of examples that each net saw this epoch \n')
                utils.double_log(f_log, np.array2string(
                    accum_min_index) + '\n')
                utils.double_log(
                    f_log, 'total number of examples that each net saw this epoch, per class\n')
                utils.double_log(
                    f_log, '\n' + np.array2string(hist_classes) + '\n')
                if epoch % period_val_acc == 0:
                    validation_accuracy = val_test(n_step, mode='val')
                    validation_accuracies.append(validation_accuracy)
                    if validation_accuracy >= np.max(validation_accuracies):
                        best_step = n_step
                        test_saver.save(
                            sess, os.path.join(ckpt_path, 'test/model.ckpt'), global_step=n_step)
                        continue
                    elif not any(x < validation_accuracy for x in validation_accuracies[:-patience]) and epoch > 10:
                        break
                    else:
                        continue
                else:
                    continue

        utils.double_log(f_log, "Optimization Finished!\n")
        test_saver.save(
            sess, os.path.join(ckpt_path, 'test/model.ckpt'), global_step=n_step)
        val_test(n_step + 1, mode='test')
        restorers.restore_all_weights(sess, os.path.join(
            ckpt_path, 'test/model.ckpt-' + str(best_step)))
        val_test(n_step + 2, mode='test')
        f_log.close()
        summary_writer.close()


def main():
    exp_id_prefix = 'DMCL'
    # creates experiment_id and makedirs for checkpoints and logs
    exp_id = utils.create_folders(prefix=exp_id_prefix, dry_run=args.dryrun,
                                  dset=args.dset, eval_mode=args.eval_mode)
    # gets the names of videos for train/val/test set
    train_fnames, val_fnames, test_fnames = utils.get_dset_filenames_step2(
        dset=args.dset, argsmini=args.argsmini, notes=args.notes)
    n_classes = utils.get_n_classes(args.dset)
    train(exp_id, train_fnames, val_fnames, test_fnames, n_classes)


if __name__ == '__main__':
    main()
