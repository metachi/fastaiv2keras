from keras import backend as K
import random

class CenterCrop():
    def __init__(self, sz, preprocess=None):
        self.sz = sz
        self.preprocess = preprocess
    def __call__(self, img):
        if self.preprocess:
            img = self.preprocess(img)
        if K._image_data_format == 'channels_last':
            r,c,_= img.shape
            return img[int((r-self.sz)/2):int((r-self.sz)/2)+self.sz, int((c-self.sz)/2):int((c-self.sz)/2)+self.sz]
        else:
            _,r,c= img.shape
            return img[:, int((r-self.sz)/2):int((r-self.sz)/2)+self.sz, int((c-self.sz)/2):int((c-self.sz)/2)+self.sz]

class RandCrop():
    def __init__(self, sz, preprocess=None):
        self.sz = sz
        self.preprocess = preprocess
    def __call__(self, img):
        if self.preprocess:
            img = self.preprocess(img)
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