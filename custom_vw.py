
# coding: utf-8

# In[1]:

from __future__ import division

import os
import codecs
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
from subprocess import call


# In[15]:

def holdout_cv_vw(train_file_vw,
                  train_size=10000,
                  train_params='',
                  to_dir = 'cv',
                  scoring=mean_absolute_error,
                  make_sets = True,
                  mute = False):
    
    """ This function performs holdout cross validation with wovpalwabbit.
        It takes a path to training data in vw format as an argument and
        creates train and validation sets, as well a as model and prediction files
        in subdirectory to_dir. In the end, the model is retrained on both
        train and test data and saved as file. Temporaryfiles are removed.   

    Args:
        train_file_vw: path to training data in vw format
        train_size:    size of train set, the rest is use as validation set
        train_params:  string of options for vowpal training 
        to_dir:        name of the folder to save models to
        scoring:       pass function that returns score list of predictions and list of targets
        make_sets:     if True, create train and validation sets; else use existing ones
        mute:          print bash commands or not

    Returns:
        a score on holdout set
    """
    
    path = '/'.join(train_file_vw.split('/')[:-1]) + '/' + to_dir + '/'
    
    if make_sets:
        if not os.path.exists(path):
            os.makedirs(path)
        # need codecs.open for texts with Cyrillic characters
        # without them, use simple open and drop 'utf-8'
        y = []
        with codecs.open(train_file_vw, 'r', 'utf-8') as data,              codecs.open(path+'train.vw', 'w', 'utf-8') as train,              codecs.open(path+'valid.vw', 'w', 'utf-8') as valid,             open(path+'train_y.txt', 'w') as train_y,             open(path+'valid_y.txt', 'w') as valid_y:
            
            for i, line in enumerate(data):
                if i < train_size:
                    train.write(line)
                    train_y.write(line.split(' ', 1)[0]+'\n')
                else:
                    valid.write(line)
                    valid_y.write(line.split(' ', 1)[0]+'\n')

    
    # train a model on validation training set
    command = 'vw -c -k -d ' + path+'train.vw -f ' + path + 'model.vw ' + train_params
    if not mute: print command
    command = command.split()
    call(command)

    # make a prediction on train set
    command = 'vw -i ' + path+'model.vw -t -d ' + path + 'train.vw -p ' + path + 'train_p.txt'
    if not mute: print command
    command = command.split()
    call(command)
  
    # make a prediction on holdout set
    command = 'vw -i ' + path+'model.vw -t -d ' +path+'valid.vw -p ' + path+'valid_p.txt'
    if not mute: print command
    command = command.split()
    call(command)

    with open(path+'train_p.txt') as f:
        train_p = [float(label) for label in f.readlines()]
    os.remove(path+'train_p.txt')
    with open(path+'train_y.txt') as f:
        train_y = [float(label) for label in f.readlines()]
    train_score = scoring(train_p, train_y)
    
    with open(path+'valid_p.txt') as f:
        valid_p = [float(label) for label in f.readlines()]
    os.remove(path+'valid_p.txt')
    with open(path+'valid_y.txt') as f:
        valid_y = [float(label) for label in f.readlines()]
    valid_score = scoring(valid_p, valid_y)
    
    # train a model on full training set
#     command = 'vw -c -k -d ' + train_file_vw + ' -f ' + path+'model.vw ' + train_params
#     if not mute: print command
#     command = command.split()
#     call(command)    
#     model = path+'model.vw'

#     os.remove(path + 'train.vw')
#     os.remove(path + 'valid.vw')
#     os.remove(path)

    return train_score, valid_score


# In[14]:

train_file_vw = '../../data/data.hw8/habr_train.vw'

train_length = 120000
test_fraction = 1/4
train_size = int(train_length * (1-test_fraction))

train_params = '--passes 3'
to_dir = 'tmp'
scoring = mean_absolute_error

# holdout_cv_vw(train_file_vw = train_file_vw,
#               train_size = train_size,
#               train_params = train_params,
#               to_dir = to_dir,
#               scoring = scoring,
#               make_sets = True,
#               mute = False)


# In[4]:

def pred_vw(test_file_vw, model):
    
    """ This function takes a file path to vowpawabbit model
        and a path to test data file in vw format, returns array of averaged
        predictions
        

    Args:
        test_file_vw: path to test data in vw format
        model:        path to vw model

    Returns:
        array of predictions, averaged over all models
    """
    
    preds=[]

    path = '/'.join(model.split('/')[:-1]) + '/'

    command = 'vw -i ' + model + ' -t -d ' + test_file_vw + ' -p ' + path + 'tmp.txt'
    print command
    command = command.split()
    call(command)

    with open(path+'tmp.txt') as pred_file:
        pred = [float(label) for label in pred_file.readlines()]

    os.remove(path+'tmp.txt')
    return pred


# In[6]:

# use for randomized holdout

# def holdout_cv_vw(train_file_vw,
#                   train_params='',
#                   to_dir = 'cv',
#                   scoring=mean_absolute_error,
#                   test_size=0.33,
#                   random_state=42,
#                   shuffle=False):
    
#     """ This function performs holdout cross validation with wovpalwabbit.
#         It takes a path to training data in vw format as an argument and
#         creates train and validation sets, as well a as model and prediction files
#         in subdirectory to_dir. In the end, the model is retrained on both
#         train and test data and saved as file. Temporaryfiles are removed.   

#     Args:
#         train_file_vw: path to training data in vw format
#         train_params:  string of options for vowpal training 
#         to_dir:        name of the folder to save models to
#         scoring:       pass function that returns score list of predictions and list of targets
#         test_size:     size of holdout, same as rest_size for sklearn.train_test_split
#         random_state:
#         shuffle:

#     Returns:
#         a score on holdout set
#         a path to the model
#     """
    
#     path = '/'.join(train_file_vw.split('/')[:-1]) + '/' + to_dir + '/'
#     if not os.path.exists(path):
#         os.makedirs(path)
    
#     # need codecs.open for texts with Cyrillic characters
#     # without them, use simple open and drop 'utf-8'
#     with codecs.open(train_file_vw, 'r', 'utf-8') as f:
#         data = f.readlines()
#     y = [float(s.split(' ', 1)[0]) for s in data]

#     train, valid, _, valid_y = train_test_split(data, y, test_size=test_size, random_state=random_state)

#     scores = []
#     models = []

#     with codecs.open(path+'train.vw', 'w', 'utf-8') as vw_train:
#         for line in (train):
#             vw_train.write(line)
#     with codecs.open(path+'valid.vw', 'w', 'utf-8') as vw_train:
#         for line in (valid):
#             vw_train.write(line)

#     #train a model
#     command = 'vw -d ' + path+'train.vw -f ' + path+'model.vw ' + train_params
#     print command
#     command = command.split()
#     call(command)
  
#     # make a prediction
#     command = 'vw -i ' + path+'model.vw -t -d ' + \
#                path+'valid.vw -p ' + path+'pred.txt'
#     print command
#     command = command.split()
#     call(command)

#     with open(path+'pred.txt') as pred_file:
#         pred = [float(label) for label in pred_file.readlines()]
            
#     scores.append(scoring(valid_y, pred))
#     models.append(path+'model.vw')
        
# #         os.remove(path+'model'+str(k)+'.vw')
#     os.remove(path+'pred.txt')

#     os.remove(path + 'train.vw')
#     os.remove(path + 'valid.vw')
#     del data

#     return scores, models




def kfold_cv_vw(train_file_vw,
                train_params='',
                to_dir = 'cv',
                scoring=mean_absolute_error,
                folds=5,
                random_state=42,
                shuffle=False):
    
    """ This function performs kfold cross validation with wovpalwabbit.
        It takes a path to training data in vw format as an argument and
        creates train and validation sets, as well as k model files
        and k prediction files in to_dir subdirectory. Only model files are kept,
        the rest are deleted in the end.        

    Args:
        train_file_vw: path to training data in vw format
        train_params:  string of options for vowpal training 
        to_dir:        name of the folder to save models to
        scoring:       pass function that returns score list of predictions and list of targets
        folds:         number of folds, default is 5
        random_state:
        shuffle:

    Returns:
        list of k scores on k folds
        list of paths to k models
    """
    
    path = '/'.join(train_file_vw.split('/')[:-1]) + '/' + to_dir + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # need codecs.open for texts with Cyrillic characters
    # without them, use simple open and drop 'utf-8'
    with codecs.open(train_file_vw, 'r', 'utf-8') as f:
        data = f.readlines()
    y = [float(s.split(' ', 1)[0]) for s in data]

    kf = KFold(n_splits=folds, random_state=random_state, shuffle=shuffle)
    scores = []
    models = []

    for k, split in enumerate(tqdm_notebook(kf.split(data), total=folds)):

        # create train and validation sets for a given split
        train = [data[i] for i in split[0]]
        valid = [data[i] for i in split[1]]

        valid_y = [y[i] for i in split[1]]

        with codecs.open(path+'train.vw', 'w', 'utf-8') as vw_train:
            for line in (train):
                vw_train.write(line)
        with codecs.open(path+'valid.vw', 'w', 'utf-8') as vw_train:
            for line in (valid):
                vw_train.write(line)

        #train a model
        command = 'vw -d ' + path+'train.vw -f ' + path+'model'+str(k)+'.vw ' + train_params
        command = command.split()
        call(command)
  
        # make a prediction
        command = 'vw -i ' + path+'model'+str(k)+'.vw -t -d ' +                    path+'valid.vw -p ' + path+'pred'+str(k)+'.txt'
        command = command.split()
        call(command)

        with open(path+'pred'+str(k)+'.txt') as pred_file:
            pred = [float(label) for label in pred_file.readlines()]
            
        scores.append(scoring(valid_y, pred))
        models.append(path+'model'+str(k)+'.vw')
        
#         os.remove(path+'model'+str(k)+'.vw')
        os.remove(path+'pred'+str(k)+'.txt')

    os.remove(path + 'train.vw')
    os.remove(path + 'valid.vw')
    del data

    return scores, models




def pred_ens_vw(test_file_vw, models_paths):
    
    """ This function takes a list of file paths to vowpawabbit models
        and a path to test data file in vw format, returns array of averaged
        predictions
        

    Args:
        test_file_vw: path to test data in vw format
        model_paths:  list of paths to vw models 

    Returns:
        array of predictions, averaged over all models
    """
    
    preds=[]

    for model in tqdm_notebook(models_paths, total=len(models_paths)):
        path = '/'.join(model.split('/')[:-1]) + '/'

        command = 'vw -i ' + model + ' -t -d ' + test_file_vw + ' -p ' + path+'tmp.txt'
        command = command.split()
        call(command)

        with open(path+'tmp.txt') as pred_file:
            pred = [float(label) for label in pred_file.readlines()]
        preds.append(pred) 
        
        os.remove(path+'tmp.txt')

    return np.array(preds).mean(axis=0)


# In[25]:

# scores, models = kfold_cv_vw(train_file_vw, train_params=train_params,
#                              scoring=scoring, folds=folds,
#                              random_state=random_state, shuffle=True)


# In[58]:

# with codecs.open('../../data/data.hw8/cv/valid.vw', 'r', 'utf-8') as f:
#     data = f.readlines()
# valid_y = [float(s.split(' ', 1)[0]) for s in data]
# del data

# command = 'vw -d ../../data/data.hw8/cv/train.vw -f ../../data/data.hw8/cv/model.vw' + \
# ' --passes 3 --cache_file ../../data/data.hw8/cv/train.cache'
# call(command.split())
# !vw -d ../../data/data.hw8/cv/train.vw -f ../../data/data.hw8/cv/model.vw \
# --passes 3 --cache_file ../../data/data.hw8/cv/train.cache


# In[59]:

# command = 'vw -i ../../data/data.hw8/cv/model.vw -t -d ../../data/data.hw8/cv/valid.vw ' + \
#           '-p ../../data/data.hw8/cv/pred.txt'
# call(command.split())

# with open('../../data/data.hw8/cv/pred.txt') as pred_file:
#     pred = [float(label) for label in pred_file.readlines()]


# In[60]:

# mean_absolute_error(pred, valid_y)


# In[61]:

# check stdout here

# path = '../../data/data.hw8/cv/'
# command = 'vw -d ../../data/data.hw8/cv/train.vw -f ../../data/data.hw8/cv/model.vw'
# p = Popen(command.split(), stdout=PIPE)
# print p.communicate()[0]


# In[ ]:



