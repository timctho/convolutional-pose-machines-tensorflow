import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph

from config import FLAGS


def freeze_model():
    model_path_suffix = os.path.join(FLAGS.network_def,
                                     'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                     'joints_{}'.format(FLAGS.num_of_joints),
                                     'stages_{}'.format(FLAGS.cpm_stages),
                                     'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                      FLAGS.lr_decay_step)
                                     )
    model_save_dir = os.path.join('models',
                                  'weights',
                                  model_path_suffix)
    model_path = os.path.join(model_save_dir, FLAGS.model_path)
    model_path = 'models/weights/cpm_hand'

    # Load graph and dump to protobuf
    meta_graph = tf.train.import_meta_graph(model_path + '.meta')
    tf.train.write_graph(tf.get_default_graph(), 'frozen_models/', 'graph_proto.pb')

    output_graph_path = os.path.join('frozen_models', '{}_frozen.pb'.format('cpm_hand'))
    freeze_graph(input_graph='frozen_models/graph_proto.pb',
                 input_saver='',
                 input_checkpoint=model_path,
                 output_graph=output_graph_path,
                 output_node_names=FLAGS.output_node_names,
                 restore_op_name='save/restore_all',
                 clear_devices=True,
                 initializer_nodes='',
                 variable_names_blacklist='',
                 input_binary=False,
                 filename_tensor_name='save/Const:0')


if __name__ == '__main__':
    freeze_model()
