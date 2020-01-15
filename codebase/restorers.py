import tensorflow as tf


def restore_all_weights(sess, checkpoint_filename, rm_global_step=False):
    if rm_global_step:
        variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_to_restore = [
            x for x in variables_to_restore if 'global_step' not in x.name]
        restorer = tf.train.Saver(variables_to_restore)
    else:
        restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_filename)
