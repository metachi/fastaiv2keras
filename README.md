# fastaiv2keras
This is an implementation of the fastai part1 v2 course in Keras
lesson1 and lesson1-finetune2 jupyter notebooks go through the dogs and cats dataset

- lesson1 uses a finetune function that simply freezes early layers and makes the fc layer output predictions for 2 classes
- lesson1-finetune2 uses finetune2 function.  It uses a few extra layers to enhance the model.  It adds average and max pooling layers and concatenates them and then follows them with batchnorm, dropout, and dense layers.  Note: that our function to find the optimal learning rate (LR_FIND) does not seem to work quite as well when we use finetune2.
