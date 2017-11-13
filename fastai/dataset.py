from .transform import *
from enum import IntEnum
from keras.preprocessing.image import ImageDataGenerator

class ImageClassifierData:
    @classmethod
    def from_paths(self, path, bs=64, tfms=None, trn_name='train', val_name='valid', sz=None, **kwargs):
        '''
        @args:
            path: directory where data resides
            bs: batch size
            tfms: transformations to perform tuple of Keras ImageDataGenerator's (train,validation)
            trn_name: subfolder name of training data (path+trn_name)
            val_name: subfolder name of validation data (path+val_name)
            **kwargs to pass to keras flow_from_directory function
        @returns:
            a tuple of image batch generators (train, validation)
        '''
        if sz == None:
            try:
                sz = tfms[0].sz
            except AttributeError:
                raise Exception('sz must be set through tfms_from_model or passed as an argument')
        
        if tfms == None: tfms = ImageDataGenerator(), ImageDataGenerator()
        train = tfms[0].flow_from_directory(path+trn_name, target_size=(sz,sz),
                class_mode='categorical', shuffle=True, batch_size=bs, **kwargs)
        valid = tfms[0].flow_from_directory(path+val_name, target_size=(sz,sz),
                class_mode='categorical', shuffle=True, batch_size=bs, **kwargs)
        return train,valid
    
class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    
def tfms_from_model(model, sz, crop_type=CropType.CENTER, **kwargs):
    '''
    @args:
        model: architecture
        target" size image
        crop_type: type of cropping
        **kwargs: other arguments to be passed into ImageDataGenerator
    @returns:
        a tuple of train and validation Keras ImageDataGenerators
    '''
    crop = CenterCrop(sz)
    if crop_type == CropType.RANDOM:
        crop = RandCrop(sz)
    elif crop_type == CropType.NO:
        crop = None
    
    train_gen = ImageDataGenerator(preprocessing_function=crop, **kwargs)
    val_gen = ImageDataGenerator(preprocessing_function=CenterCrop(sz))
    train_gen.sz = sz
    val_gen.sz = sz
    return train_gen, val_gen