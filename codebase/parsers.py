import tensorflow as tf


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    with tf.name_scope(scope, 'distort_color', [image]):
        def fn1(image):
            image = tf.image.random_brightness(image, max_delta=32.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            return image

        def fn2(image):
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0)
            return image

        def fn3(image):
            image = fn1(image)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def fn4(image):
            image = fn2(image)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            return image

        def fn5(image):
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = fn1(image)
            return image

        def fn6(image):
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0)
            return image

        image = tf.case({tf.logical_and(fast_mode, tf.less(color_ordering, 2)): lambda: fn1(image),
                         tf.logical_and(fast_mode, tf.greater_equal(color_ordering, 2)): lambda: fn2(image),
                         tf.logical_and(tf.math.logical_not(fast_mode), tf.equal(color_ordering, 0)): lambda: fn3(image),
                         tf.logical_and(tf.math.logical_not(fast_mode), tf.equal(color_ordering, 1)): lambda: fn4(image),
                         tf.logical_and(tf.math.logical_not(fast_mode), tf.equal(color_ordering, 2)): lambda: fn5(image),
                         tf.logical_and(tf.math.logical_not(fast_mode), tf.equal(color_ordering, 3)): lambda: fn6(image)}, exclusive=True)
        return tf.clip_by_value(image, 0.0, 255.0)


def sample_frames_step2(filenames2, video_len, n_frames):
    partitions_len = tf.divide(video_len, n_frames)
    index = tf.range(n_frames)
    index = tf.map_fn(lambda x: tf.random_uniform(
        [], minval=tf.cast(tf.ceil(tf.cast(x, tf.float64) * partitions_len), tf.int32), maxval=tf.cast(tf.ceil((tf.cast(x, tf.float64) + 1) * partitions_len), tf.int32), dtype=tf.int32), index)

    return tf.gather(filenames2[0], index), tf.gather(filenames2[1], index), tf.gather(filenames2[2], index)


def _parse_mult_frame_s2(example_proto, n_frames=8, rescale=False, distort=True):
    # tfrecords stored in _paths_
    # have all frames' paths for the three modalities
    # this function returns n_frames of each modality

    with tf.device('/cpu:0'):
        context_features = {"label": tf.FixedLenFeature(
            [], dtype=tf.int64), "video_id": tf.FixedLenFeature([], dtype=tf.string),
            "length": tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {
            "rgb_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "depth_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "oflow_frames": tf.FixedLenSequenceFeature([], dtype=tf.string)}

        context, features = tf.parse_single_sequence_example(
            serialized=example_proto, context_features=context_features, sequence_features=sequence_features)

        video_len = tf.cast(context['length'], tf.int32)
        rgb_filenames = features['rgb_frames']
        depth_filenames = features['depth_frames']
        oflow_filenames = features['oflow_frames']

        filenames = sample_frames_step2(
            [rgb_filenames, depth_filenames, oflow_filenames], video_len, n_frames)
        rgb_filenames = filenames[0]
        depth_filenames = filenames[1]
        oflow_filenames = filenames[2]

        def read_img(in_filenames, in_flipping_prob, in_offsets, mod='rgb'):
            frame = tf.map_fn(tf.read_file, in_filenames)
            frame = tf.map_fn(lambda x: tf.image.decode_jpeg(
                x, channels=3), frame, dtype=tf.uint8)
            frame = tf.image.resize_images(frame, [256, 256])

            frame = tf.image.extract_glimpse(
                frame, [224, 224], in_offsets, normalized=False)

            def flip(x): return tf.image.flip_left_right(x)
            frame = tf.cond(in_flipping_prob,
                            lambda: tf.map_fn(flip, frame),
                            lambda: frame)

            frame = tf.to_float(frame)

            def distort_color_(x):
                return distort_color(x, color_ordering=tf.random_uniform(
                    [], minval=0, maxval=4, dtype=tf.int32), fast_mode=tf.less(0.5, tf.random_uniform(
                        [], minval=0, maxval=1, dtype=tf.float32)))
            if mod == 'rgb' and distort:
                distort_prob = tf.random_uniform([])
                frame = tf.cond(tf.greater(distort_prob, 0.5),
                                lambda: distort_color_(frame),
                                lambda: frame)

            frame = tf.to_float(frame)
            if rescale:  # [-1, 1]
                # frame = frame / 255.0
                frame = 2 * (frame / 255.0) - 1
            return frame

        flipping_prob1 = tf.random_uniform([])
        flipping_prob = tf.greater(flipping_prob1, 0.5)

        c_x = tf.random_uniform(
            [], minval=tf.constant(-16.0), maxval=tf.constant(16.0))
        c_y = tf.random_uniform(
            [], minval=tf.constant(-16.0), maxval=tf.constant(16.0))
        offsets = tf.tile([c_x, c_y], [n_frames])
        offsets = tf.reshape(offsets, [n_frames, 2])
        offset_prob = tf.random_uniform([])
        offsets = tf.cond(tf.greater(offset_prob, 0.5),
                          lambda: offsets,
                          lambda: tf.zeros([n_frames, 2]))

        rgb_frames = read_img(rgb_filenames, flipping_prob, offsets, 'rgb')
        depth_frames = read_img(
            depth_filenames, flipping_prob, offsets, 'depth')
        oflow_frames = read_img(
            oflow_filenames, flipping_prob, offsets, 'oflow')
        label = context['label']
        video_id = tf.convert_to_tensor(context['video_id'])

    # , rgb_filenames, depth_filenames
    return rgb_frames, depth_frames, oflow_frames, label, video_id


def _parse_mult_frame_test_allmods(example_proto, rescale=False, n_frames=8):
    with tf.device('/cpu:0'):
        context_features = {"label": tf.FixedLenFeature(
            [], dtype=tf.int64), "video_id": tf.FixedLenFeature([], dtype=tf.string),
            "length": tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {
            "rgb_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "depth_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "oflow_frames": tf.FixedLenSequenceFeature([], dtype=tf.string)}

        context, features = tf.parse_single_sequence_example(
            serialized=example_proto, context_features=context_features, sequence_features=sequence_features)

        rgb_filenames = features['rgb_frames']
        depth_filenames = features['depth_frames']
        oflow_filenames = features['oflow_frames']

        def read_img(filenames, mod='rgb'):
            frame = tf.map_fn(tf.read_file, filenames)
            frame = tf.map_fn(lambda x: tf.image.decode_jpeg(
                x, channels=3), frame, dtype=tf.uint8)
            frame = tf.image.resize_images(frame, [256, 256])

            offsets = tf.zeros([n_frames, 2])
            frame = tf.image.extract_glimpse(frame, [224, 224], offsets)

            frame = tf.to_float(frame)
            if rescale:
                # frame = frame / 255.0
                frame = 2 * (frame / 255.0) - 1
            return frame

        rgb_frames = read_img(rgb_filenames, 'rgb')
        depth_frames = read_img(depth_filenames, 'depth')
        oflow_frames = read_img(oflow_filenames, 'oflow')
        label = context['label']
        video_id = tf.convert_to_tensor(context['video_id'])

    # , rgb_filenames, depth_filenames, oflow_filenames
    return rgb_frames, depth_frames, oflow_frames, label, video_id
