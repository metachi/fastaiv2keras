import keras
from keras.callbacks import Callback
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
import math
import random
import numpy as np
import matplotlib.pyplot as plt

if K.backend() == 'theano':
    if K.image_data_format() == 'channels_last':
        K.set_image_data_format('channels_first')
else:
    if K.image_data_format() == 'channels_first':
        K.set_image_data_format('channels_last')

class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''
    def __init__(self, iterations, epochs=1):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}
    def setRate(self):
        return K.get_value(self.model.optimizer.lr)
    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])
    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'][n_skip:-5], self.history['loss'][n_skip:-5])
        plt.xscale('log')



class LR_Find(LR_Updater):
    '''This callback is utilized to determine the optimal lr to be used
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai/learner.py
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self, iterations, epochs=1, min_lr=1e-05, max_lr=10, jump=6):
        '''
        iterations = dataset size / batch size
        epochs should always be 1
        min_lr is the starting learning rate
        max_lr is the upper bound of the learning rate
        jump is the x-fold loss increase that will cause training to stop (defaults to 6)
        '''
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_mult = (max_lr/min_lr)**(1/iterations)
        self.jump = jump
        super().__init__(iterations, epochs=epochs)
    def setRate(self):
        return self.min_lr * (self.lr_mult**self.trn_iterations)
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs=logs)
        try: #multiple lr's
            K.get_variable_shape(self.model.optimizer.lr)[0]
            self.min_lr = np.full(K.get_variable_shape(self.model.optimizer.lr),self.min_lr)
        except IndexError:
            pass
        K.set_value(self.model.optimizer.lr, self.min_lr)
        self.best=1e9
        self.model.save_weights('tmp.hd5') #save weights
    def on_train_end(self, logs=None):
        self.model.load_weights('tmp.hd5') #load_weights
    def on_batch_end(self, batch, logs=None):
        #check if we have made an x-fold jump in loss and training should stop
        try:
            loss = self.history['loss'][-1]
            if math.isnan(loss) or loss > self.best*self.jump:
                self.model.stop_training = True
            if loss < self.best:
                self.best=loss
        except KeyError:
            pass
        super().on_batch_end(batch, logs=logs)
        
        
        
        
class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''
    def __init__(self, iterations, cycle_len=1, cycle_mult=1, epochs=1):
        '''
        iterations = dataset size / batch size
        epochs #todo do i need this or can it accessed through self.model
        cycle_len = num of times learning rate anneals from its max to its min in an epoch
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations, epochs=epochs)
    def setRate(self):
        self.cycle_iterations += 1
        cos_out = np.cos(np.pi*(self.cycle_iterations)/self.epoch_iterations) + 1
        if self.cycle_iterations==self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        return self.max_lr / 2 * cos_out
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={}) #changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)


class SGD2(optimizers.Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        split1_layer: first middle layer (uses 2nd learning rate)
        split2_layer: first top layer (uses final/3rd learning rate)
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, split1_layer, split2_layer, lr=[0.0001, .001, .01], momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.split1 = split1_layer.weights[0].name
            self.split2 = split2_layer.weights[0].name
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @keras.optimizers.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
#         print(type(self.lr))
#         [print(type(item),item.name) for item in params]
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        grp = 0 #set layer grp to 1
        for p, g, m in zip(params, grads, moments):
            if self.split1 == p.name:
                grp = 1
            if self.split2 == p.name:
                grp = 2
#             print("lr_grp",grp)
            v = self.momentum * m - lr[grp] * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr[grp] * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def finetune(model, num_classes):
    '''removes the last layer of a nn and adds a fully-connected layer for predicting num_classes
    '''
    model.layers.pop()
    for layer in model.layers: layer.trainable = False
    last = model.output
    preds = keras.layers.Dense(num_classes, activation='softmax')(last)
    return Model(model.input, preds)

    
def finetune2(model, pool_layer, num_classes):
    '''removes the last layers of a nn and adds a fully-connected layers 
    for predicting num_classes
    
    # Arguments
        model: model to finetune
        pool_layer: pooling layer after the final conv layers
            *note this will be replaced by a AvgMaxPoolConcatenation
    '''
    model = Model(model.input, pool_layer.input)
    model.layers.pop()
    for layer in model.layers: layer.trainable = False
    last = model.output
    a = keras.layers.MaxPooling2D(pool_size=(7,7),name='maxpool')(last)
    b = keras.layers.AveragePooling2D(pool_size=(7,7),name='avgpool')(last)
    x = keras.layers.concatenate([a,b], axis = 1)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization(epsilon=1e-05, name='start_flat')(x)
    #x = keras.layers.Dropout(.25)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-05)(x)
    #x = keras.layers.Dropout(.5)(x)
    preds = keras.layers.Dense(num_classes, activation='softmax')(x)
    return Model(model.input, preds)

class CenterCrop():
    def __init__(self, sz): self.sz = sz
    def __call__(self, img):
        if K._image_data_format == 'channels_last':
            r,c,_= img.shape
            return img[int((r-self.sz)/2):int((r-self.sz)/2)+self.sz, int((c-self.sz)/2):int((c-self.sz)/2)+self.sz]
        else:
            _,r,c= img.shape
            return img[:, int((r-self.sz)/2):int((r-self.sz)/2)+self.sz, int((c-self.sz)/2):int((c-self.sz)/2)+self.sz]

class RandCrop():
    def __init__(self, sz): self.sz = sz
    def __call__(self, img):
        if K._image_data_format == 'channels_last':
            r,c,_= img.shape
            start_r = random.randint(0, r-self.sz)
            start_c = random.randint(0, c-self.sz)
            return img[start_r:start_r+self.sz, start_c:start_c+self.sz]
        else:
            _,r,c= img.shape
            start_r = random.randint(0, r-self.sz)
            start_c = random.randint(0, c-self.sz)
            return img[:, start_r:start_r+self.sz, start_c:start_c+self.sz]