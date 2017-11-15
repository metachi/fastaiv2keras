from .utils import *
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras import optimizers
from keras.models import Model, load_model
import math

model_meta = {
    ResNet50:['res4a_branch2a','fc_start'], VGG16: [0,22], VGG19:[8,6], 
    InceptionV3:[8,6], Xception:[8,6],
}

class ConvLearner(): #todo implement learner parent class
    def __init__(self, model, data, arch):
        self.data = data
        self.arch = arch
        self.model = model
    @classmethod
    def pretrained(cls, arch, data, precompute=False, finetune2_layer=None):
        #todo implement precomputed activations for faster training
        if finetune2_layer:
            model = finetune2(arch(), data[0].num_class, finetune2_layer)
        else:
            model = finetune(arch(), data[0].num_class)
        return cls(model, data, arch)
    def fit(self, lrs, n_cycle, cycle_len=None, cycle_mult=1, 
            metrics=['accuracy'], callbacks = [], workers=4, **kwargs):
        '''
        @args:
            lrs: learning rates, can pass 1 or a list of 3
            n_cycle: number of cycles (epochs if cycle_mult = 1)
            cycle_len: used to implement a cyclical learning rate with cosine annealing
            cycle_mult: used to decrease the rate of annealing by cycle_mult times every cycle
            **kwargs: to be passed to the keras fit_generator
        @returns:
            
        '''
        self.callbacks = callbacks
        #check to see if multiple lr's were passed
        if isinstance(lrs,list) and len(lrs)>1:
            #get layers where we need to split learning rates
            conv_layer = [layer for layer in self.model.layers if layer.name == model_meta[self.arch][0]][0]
            fc_layer = [layer for layer in self.model.layers if layer.name == model_meta[self.arch][1]][0]
            sgd = SGD2(conv_layer, fc_layer, lr=lrs)
        else:
            if isinstance(lrs,list): lrs=lrs[0]
            sgd = optimizers.SGD(lr=lrs, momentum=0.9)
        #if cycle_len is set
        if cycle_len:
            epochs = cycle_len * n_cycle if cycle_mult ==1 else math.ceil(cycle_len*(1-cycle_mult**n_cycle)/(1-cycle_mult))
            self.sched = LR_Cycle(math.ceil(self.data[0].samples/self.data[0].batch_size),
                                  cycle_len=cycle_len, cycle_mult=cycle_mult, epochs=epochs)
            callbacks.append(self.sched)
        else:
            epochs = n_cycle
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=metrics)
        self.model.fit_generator(self.data[0], 
                                 steps_per_epoch=math.ceil(self.data[0].samples/self.data[0].batch_size),
                                 epochs=epochs, 
                                 callbacks=callbacks, 
                                 workers=workers,
                         validation_data=self.data[1], 
                         validation_steps=math.ceil(self.data[1].samples/self.data[1].batch_size),
                                )
    def unfreeze(self):
        '''unfreeze all layers of the model'''
        for layer in self.model.layers: layer.trainable = True
    def save(self,path):
        #recompile to get around error when saving w/ multi lrs
        sgd = optimizers.SGD(lr=.08)
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        self.model.save(path)
    def load(self,path):
        self.model = load_model(path)
    def lr_find(self,jump=6,workers=4):
        self.sched = LR_Find(math.ceil(self.data[0].samples/self.data[0].batch_size),jump=jump)
        sgd = optimizers.SGD(lr=1e-05, momentum=0.9)
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd)
        self.model.fit_generator(self.data[0],
                            steps_per_epoch=math.ceil(self.data[0].samples/self.data[0].batch_size),
                    epochs=1, callbacks=[self.sched], workers=workers
                   )