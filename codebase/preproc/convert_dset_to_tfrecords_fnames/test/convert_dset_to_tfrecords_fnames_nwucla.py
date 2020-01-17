import os
import tensorflow as tf
import numpy as np
# import ipdb

# this script writes one tfrecord file per val/test clip
# each tfrecord saves 3 lists of paths, one for each modality
# each list has 8 strings, one per image of the clip.
# clip: each video is divided in 8 equal parts;
# randomly sample 1 frame from each part;
# do this 10 times, and save each clip on a different tfrecord;

data_dir = '/path_to_data/nwucla/'
rgb_depth_dset_dir = 'multiview_action'

dict_translate_actions = {1: 1, 2: 2, 3: 3,
                          4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 11: 9, 12: 10}


def get_fnames():
    recs = []
    for view in range(1, 4):
        videos = os.listdir(os.path.join(
            data_dir, rgb_depth_dset_dir, 'view_' + str(view)))
        videos = ['view_' + str(view) + '/' + x for x in videos]
        recs.extend(videos)
    return recs


def main():
    depth_dset_dir = os.path.join(data_dir, 'multiview_action/')
    oflow_dset_dir = os.path.join(data_dir, 'flow/')
    rgb_dset_dir = os.path.join(data_dir, 'multiview_action/')

    output_dir = os.path.join(
        data_dir, 'tfrecords/depth_vis/testval_tfrecords/')

    # each video is divided in 8 equal parts;
    # randomly sample 1 frame from each part;
    # do this 10 times, and save each clip on a different tfrecord;
    n_frames = 8
    # videos_ids = os.listdir(os.path.join(rgb_dset_dir, view_id))
    # add here the validation files. read json with validation files, and add

    # all videos
    videos_ids = get_fnames()
    videos_ids.sort()
    for video_id in videos_ids:
        frames_rgb = os.listdir(os.path.join(rgb_dset_dir, video_id))
        frames_rgb = [x for x in frames_rgb if '_rgb.jpg' in x]
        if not frames_rgb:
            print(video_id + " doesn't contain frames")
            continue
        frames_depth = os.listdir(os.path.join(
            depth_dset_dir, video_id))
        frames_depth = [x for x in frames_depth if 'depth_vis.jpg' in x]
        oflow_action = 'a' + \
            str(dict_translate_actions[int(video_id[8:10])]).zfill(2)
        frames_oflow = os.listdir(os.path.join(
            oflow_dset_dir, video_id[0:6], oflow_action, oflow_action + video_id[10:]))
        frames_oflow.sort()
        frames_rgb.sort()
        frames_depth.sort()
        len_vid = np.min([len(frames_rgb), len(
            frames_depth), len(frames_oflow)])
        frames_rgb = frames_rgb[:len_vid]
        frames_depth = frames_depth[:len_vid]
        frames_oflow = frames_oflow[:len_vid]
        if len_vid < 9:
            for i in range(9 - len_vid):
                frames_rgb.append(frames_rgb[-1])
                frames_depth.append(frames_depth[-1])
                frames_oflow.append(frames_oflow[-1])
            len_vid = np.min([len(frames_rgb), len(
                frames_depth), len(frames_oflow)])
        partitions_len = len_vid / 8.0
        for i in range(10):
            new_frames_rgb = []
            new_frames_oflow = []
            new_frames_depth = []
            for j in range(n_frames):
                # if np.ceil(j * partitions_len) >= np.ceil((j + 1) * partitions_len):
                #     ipdb.set_trace()
                sampled = np.random.randint(
                    np.ceil(j * partitions_len), np.ceil((j + 1) * partitions_len))
                new_frames_rgb.append(frames_rgb[sampled])
                new_frames_oflow.append(frames_oflow[sampled])
                new_frames_depth.append(frames_depth[sampled])
            rgb_frames_list = list(map(lambda x: os.path.join(
                rgb_dset_dir, video_id, x), new_frames_rgb))
            depth_frames_list = list(map(lambda x: os.path.join(
                depth_dset_dir, video_id, x), new_frames_depth))

            oflow_frames_list = list(map(lambda x: os.path.join(
                oflow_dset_dir, video_id[0:6], oflow_action, oflow_action + video_id[10:], x), new_frames_oflow))

            number_of_frames = len(rgb_frames_list)
            label = dict_translate_actions[int(video_id[8:10])] - 1
            tfrecord_name = os.path.join(
                output_dir, video_id + '_clip' + str(i + 1).zfill(2) + '.tfrecord')

            if not os.path.exists(os.path.join(output_dir, video_id[0:6])):
                os.makedirs(os.path.join(output_dir, video_id[0:6]))

            writer = tf.python_io.TFRecordWriter(tfrecord_name, options=tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP))
            ex = make_example(video_id, rgb_frames_list,
                              depth_frames_list, oflow_frames_list, label, number_of_frames)
            writer.write(ex.SerializeToString())
            writer.close()


def make_example(video_id, rgb_frames_list, depth_frames_list, oflow_frames_list, label, number_of_frames):
    ex = tf.train.SequenceExample()
    ex.context.feature["label"].int64_list.value.append(label)
    ex.context.feature["video_id"].bytes_list.value.append(
        video_id.encode())
    ex.context.feature["length"].int64_list.value.append(number_of_frames)

    rgb_frames = ex.feature_lists.feature_list["rgb_frames"]
    for frame in rgb_frames_list:
        rgb_frames.feature.add().bytes_list.value.append(frame.encode())

    depth_frames = ex.feature_lists.feature_list["depth_frames"]
    for frame in depth_frames_list:
        depth_frames.feature.add().bytes_list.value.append(frame.encode())

    oflow_frames = ex.feature_lists.feature_list["oflow_frames"]
    for frame in oflow_frames_list:
        oflow_frames.feature.add().bytes_list.value.append(frame.encode())

    return ex


if __name__ == '__main__':
    main()
