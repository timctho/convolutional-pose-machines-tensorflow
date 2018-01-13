from abc import ABCMeta, abstractmethod, abstractproperty

class CPM(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, input_size, heatmap_size, stages, joints, img_type='RGB'):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        pass



