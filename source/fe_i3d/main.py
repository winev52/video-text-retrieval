import datetime
import pickle
import cv2
import os.path as path
import numpy as np
import tensorflow as tf

import i3d

_BATCH_SIZE = 4
_IMAGE_SIZE = 224
_VIDEO_FRAMES = 64
_IN_DIR = '../../data/10vid'
_CAPS_PATH = '../../data/msvd_video_caps.pkl'
_OUT_DIR = '../../data/rgb_i3d'
_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

tf.flags.DEFINE_string('model', 'rgb_imagenet', 'rgb, rgb600, rgb_imagenet, flow, flow_imagenet')
tf.flags.DEFINE_string('in_dir', _IN_DIR, 'dir of input videos')
tf.flags.DEFINE_string('out_dir', _OUT_DIR, 'dir of output features')
tf.flags.DEFINE_integer('batch_size', _BATCH_SIZE, 'batch size')
tf.flags.DEFINE_boolean('overwrite', False, 'overwrite if exist, otherwise, ignore')
tf.logging.set_verbosity(tf.logging.INFO)
_FLAGS = tf.flags.FLAGS

def main(unused_args):
    # get app parameters
    model_name = _FLAGS.model
    out_dir = _FLAGS.out_dir
    is_opticalflow = model_name in ['flow', 'flow_imagenet']

    # get available video names
    video_ids = _get_video_ids()
    features, in_ids, iterator, video_ids_placeholder, variable_map = \
        _get_flow_model(model_name, video_ids) if is_opticalflow else _get_RBG_model(model_name, video_ids)

    rgb_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    # starting session
    n_files = len(video_ids)
    print("total videos: ", n_files)
    n_processed = 0
    with tf.Session() as sess:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[model_name])
        tf.logging.info('checkpoint restored')
        
        sess.run(iterator.initializer, feed_dict={video_ids_placeholder: video_ids})    
        while True:
            try:
                out_features, out_ids = sess.run([features, in_ids])
            except tf.errors.OutOfRangeError: # when reach end of dataset
                break
                
            for i in range(len(out_ids)):
                np.save(path.join(out_dir, out_ids[i].decode()), out_features[i], allow_pickle=False)
                
            n_processed += len(out_ids)
            tf.logging.info("{datetime:%Y-%m-%d %H:%M:%S} Processed {n_processed:d}/{n_files:d}".format(
                n_files=n_files, n_processed=n_processed, datetime=datetime.datetime.now()))
            tf.logging.info("--> {0}".format(b', '.join(out_ids)))


def _get_RBG_model(model_name, video_ids):

    # load parameter
    batch_size = _FLAGS.batch_size

    # reset graph
    tf.reset_default_graph()

    # setup dataset
    file_names_placeholder = tf.placeholder(video_ids.dtype, video_ids.shape)
    dataset = tf.data.Dataset.from_tensor_slices(file_names_placeholder)
    # preprocessing by loading video
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_rgb_function, 
                                                    [filename],
                                                    [tf.float32, filename.dtype])))
    # batching up and get the iterator
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    rgb_input, vid = iterator.get_next()
    # set shape of the input
    rgb_input.set_shape([batch_size, _VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])

    # setup the model
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(final_endpoint='Mixed_5c')
        rgb_features, _ = rgb_model(rgb_input, is_training=False)
        rgb_features = tf.nn.avg_pool3d(rgb_features, ksize=[1, 2, 7, 7, 1],
                                strides=[1, 1, 1, 1, 1], padding='VALID')
        rgb_features = tf.reduce_mean(rgb_features, axis=1)
        rgb_features = tf.squeeze(rgb_features, axis=[1, 2])

    # get the variables that will be loaded from pre-train
    variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            if model_name == 'rgb600':
                variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
            else:
                variable_map[variable.name.replace(':0', '')] = variable

    return rgb_features, vid, iterator, file_names_placeholder, variable_map

def _get_flow_model(model_name, video_ids):

    # load parameter
    batch_size = _FLAGS.batch_size

    # reset graph
    tf.reset_default_graph()

    # setup dataset
    file_names_placeholder = tf.placeholder(video_ids.dtype, video_ids.shape)
    dataset = tf.data.Dataset.from_tensor_slices(file_names_placeholder)
    # preprocessing by loading video
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_opticalflow_function, 
                                                    [filename],
                                                    [tf.float32, filename.dtype])))
    # batching up and get the iterator
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    flow_input, vid = iterator.get_next()
    # set shape of the input
    flow_input.set_shape([batch_size, _VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2])

    # setup the model
    with tf.variable_scope('Flow'):
        flow_model = i3d.InceptionI3d(final_endpoint='Mixed_5c')
        flow_features, _ = flow_model(flow_input, is_training=False)
        flow_features = tf.nn.avg_pool3d(flow_features, ksize=[1, 2, 7, 7, 1],
                                strides=[1, 1, 1, 1, 1], padding='VALID')
        flow_features = tf.reduce_mean(flow_features, axis=1)
        flow_features = tf.squeeze(flow_features, axis=[1, 2])

    # get the variables that will be loaded from pre-train
    variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
            variable_map[variable.name.replace(':0', '')] = variable

    return flow_features, vid, iterator, file_names_placeholder, variable_map

def _get_video_ids():
    out_dir = _FLAGS.out_dir
    in_dir = _FLAGS.in_dir
    overwrite = _FLAGS.overwrite

    with open(_CAPS_PATH, 'rb') as f:
        np_data = pickle.load(f)
        video_ids = np_data[:,0]
        video_ids = np.unique(video_ids)

    video_ids = np.array([vid for vid in video_ids 
                        if path.isfile(path.join(in_dir, vid +".avi"))
                        and not (overwrite and path.isfile(path.join(out_dir, vid + ".npy")))])

    return video_ids

def _transform_frame_rgb(bgr_frame):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resize_rgb = cv2.resize(rgb_frame, dsize=(_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    # normalize to [-1,1]
    norm_rgb = resize_rgb / 127.5 - 1
    # transform data using (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # norm_rgb = (norm_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return norm_rgb

def _read_rgb_function(video_id):
    video_id = video_id.decode()
    file_path = path.join(_FLAGS.in_dir, video_id + ".avi")
    frames = list()

    cap = cv2.VideoCapture(file_path)
    assert cap.isOpened(), 'Cannot open file {0}'.format(video_id)

    # get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # if total frames is less than desired, loop the video
    if total_frames <= _VIDEO_FRAMES:
        for i in range(_VIDEO_FRAMES):
            read_ok, bgr_frame = cap.read()
            if not read_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_ok, bgr_frame = cap.read()

            transformed = _transform_frame_rgb(bgr_frame)
            frames.append(transformed)
    else: # randomly sample from video
        chosen_frames = np.random.choice(total_frames, _VIDEO_FRAMES, replace=False)
        chosen_frames.sort()
        for i in range(_VIDEO_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frames[i])
            read_ok, bgr_frame = cap.read()

            assert read_ok, "cannot read frame {0}".format(chosen_frames[i])

            transformed = _transform_frame_rgb(bgr_frame)
            frames.append(transformed)

    cap.release()
        
    return np.array(frames, dtype=np.float32), video_id

def _transform_frame_flow(bgr_frame):
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, dsize=(_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    ## all zeros frame cause error in optical flow calculation
    # set to 255 may fix it
    resized_frame[0][0] = 255
    return resized_frame

def _clip_normalize_flow(np_arr):
    # clip to [-20, 20]
    clipped_arr = np.clip(np_arr, -20, 20)

    # normalize to [-1,1]
    norm_frame = clipped_arr / 20.0
    return norm_frame

def _read_opticalflow_function(video_id):
    video_id = video_id.decode()
    # print(filename)
    file_path = path.join(_FLAGS.in_dir, video_id + ".avi")
    flows = list()

    cap = cv2.VideoCapture(file_path)
    assert cap.isOpened(), 'Cannot open file {0}'.format(video_id)

    # get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # setup optical flow calculator
    # of = cv2.optflow.createOptFlow_PCAFlow()
    of = cv2.createOptFlow_DualTVL1()

    # differ from normal video,  need 2 frames to form a flow
    # if total frames is less than desired, loop the video
    if total_frames - 1 <= _VIDEO_FRAMES:
        # print("loop", total_frames)
        read_ok, prev_frame = cap.read() # first frame
        prev_frame = _transform_frame_flow(prev_frame)
        for i in range(_VIDEO_FRAMES):
            read_ok, cur_frame = cap.read()
            if not read_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_ok, cur_frame = cap.read()

            cur_frame = _transform_frame_flow(cur_frame)
            flow = of.calc(prev_frame, cur_frame, None)
            flow = _clip_normalize_flow(flow)

            flows.append(flow)
            prev_frame = cur_frame
    else: # randomly sample from video
        # print("rand", total_frames)
        chosen_frames = np.random.choice(total_frames - 1, _VIDEO_FRAMES, replace=False)
        chosen_frames.sort()

        for i in range(_VIDEO_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frames[i])

            # read 2 consecutive frame
            read_ok, prev_frame = cap.read()
            assert read_ok, "cannot read frame {0}".format(chosen_frames[i])
            read_ok, cur_frame = cap.read()
            assert read_ok, "cannot read frame {0}".format(chosen_frames[i])

            prev_frame = _transform_frame_flow(prev_frame)
            cur_frame = _transform_frame_flow(cur_frame)
            flow = of.calc(prev_frame, cur_frame, None)
            flow = _clip_normalize_flow(flow)

            flows.append(flow)

    cap.release()

    return np.array(flows, dtype=np.float32), video_id

if __name__ == "__main__":
    tf.app.run(main)
