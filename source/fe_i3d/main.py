import datetime
import pickle
import cv2
import os.path as path
import numpy as np
import tensorflow as tf

import i3d

_BATCH_SIZE = 8
_IMAGE_SIZE = 224
_VIDEO_FRAMES = 64
_VIDEO_DIR = '../../data/10vid'
_CAPS_PATH = '../../data/msvd_video_caps.pkl'
_OUTPUT_DIR = '../../data/i3d'
_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

tf.flags.DEFINE_string('model', 'rgb_imagenet', 'rgb, rgb600, rgb_imagenet')
tf.flags.DEFINE_string('video_dir', _VIDEO_DIR, 'dir of input videos')
tf.flags.DEFINE_string('output_dir', _OUTPUT_DIR, 'dir of output features')
tf.logging.set_verbosity(tf.logging.INFO)
_FLAGS = tf.flags.FLAGS

def main(unused_args):
    # get app parameters
    model_name = _FLAGS.model
    output_dir = _FLAGS.output_dir

    # get available video names
    filenames = _read_video_caps()
    tf.reset_default_graph()

    # setup dataset
    filenames_placeholder = tf.placeholder(filenames.dtype, filenames.shape)
    dataset = tf.data.Dataset.from_tensor_slices(filenames_placeholder)
    # preprocessing by loading video
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_video_function, 
                                                    [filename],
                                                    [tf.float32, filename.dtype])))
    # batching up and get the iterator
    dataset = dataset.batch(_BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    rgb_input, video_ids = iterator.get_next()
    # set shape of the input
    rgb_input.set_shape([_BATCH_SIZE, _VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])

    # setup the model
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(final_endpoint='Mixed_5c')
        rgb_features, _ = rgb_model(rgb_input, is_training=False)
        rgb_features = tf.nn.avg_pool3d(rgb_features, ksize=[1, 2, 7, 7, 1],
                                strides=[1, 1, 1, 1, 1], padding='VALID')
        rgb_features = tf.reduce_mean(rgb_features, axis=1)
        rgb_features = tf.squeeze(rgb_features, axis=[1, 2])

    # get the variables that will be loaded from pre-train
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            if model_name == 'rgb600':
                rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
            else:
                rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    # starting session
    n_files = len(filenames)
    n_processed = 0
    with tf.Session() as sess:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[model_name])
        tf.logging.info('RGB checkpoint restored')
        
        sess.run(iterator.initializer, feed_dict={filenames_placeholder: filenames})    
        while True:
            try:
                out_features, out_ids = sess.run([rgb_features, video_ids])
            except tf.errors.OutOfRangeError: # when reach end of dataset
                break
                
            for i in range(len(out_ids)):
                np.save(path.join(output_dir, out_ids[i].decode()), out_features[i], allow_pickle=False)
                
            n_processed += len(out_ids)
            tf.logging.info("{datetime:%Y-%m-%d %H:%M:%S} Processed {n_processed:d}/{n_files:d}".format(
                n_files=n_files, n_processed=n_processed, datetime=datetime.datetime.now()))
            tf.logging.info("--> {0}".format(b', '.join(out_ids)))


def _read_video_caps():
    with open(_CAPS_PATH, 'rb') as f:
        np_data = pickle.load(f)
        filenames = np_data[:,0]
        filenames = np.unique(filenames)

    filenames = np.array([f for f in filenames if path.isfile(path.join(_VIDEO_DIR, f))])

    return filenames

def _transform_frame(bgr_frame):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resize_rgb = cv2.resize(rgb_frame, dsize=(_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    # normalize to [0,1]
    norm_rgb = resize_rgb / 255.0
    # transform data using (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    norm_rgb = (norm_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return norm_rgb

def _read_video_function(filename):
    filename = filename.decode()
    file_path = path.join(_FLAGS.video_dir, filename)
    frames = list()

    cap = cv2.VideoCapture(file_path)
    assert cap.isOpened(), 'Cannot open file {0}'.format(filename)

    # get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # if total frames is less than desired, loop the video
    if total_frames <= _VIDEO_FRAMES:
        for i in range(_VIDEO_FRAMES):
            read_ok, bgr_frame = cap.read()
            if not read_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_ok, bgr_frame = cap.read()

            transformed = _transform_frame(bgr_frame)
            frames.append(transformed)
    else: # randomly sample from video
        chosen_frames = np.random.choice(total_frames, _VIDEO_FRAMES, replace=False)
        chosen_frames.sort()
        for i in range(_VIDEO_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frames[i])
            read_ok, bgr_frame = cap.read()

            assert read_ok, "cannot read frame {0}".format(chosen_frames[i])

            transformed = _transform_frame(bgr_frame)
            frames.append(transformed)

    cap.release()
        
    return np.array(frames, dtype=np.float32), filename[:-4]

if __name__ == "__main__":
    tf.app.run(main)