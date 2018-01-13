import os
import argparse

from config import FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
args = parser.parse_args()

if __name__ == '__main__':

    """ Create dirs for saving models and logs
    """
    if args.train and args.test:
        raise ValueError('Can\'t open train and test log and same time.')
    elif args.train:
        log_type = 'train'
    else:
        log_type = 'test'
    log_path = os.path.join('models',
                            'logs',
                            FLAGS.network_def,
                            'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                            'joints_{}'.format(FLAGS.num_of_joints),
                            'stages_{}'.format(FLAGS.cpm_stages),
                            'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step),
                            log_type
                            )
    print('Show tensorboard log in [{}]'.format(log_path))
    os.system('tensorboard --logdir={}'.format(log_path))
