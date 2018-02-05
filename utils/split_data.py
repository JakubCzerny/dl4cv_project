'''
    Author: Jakub Czerny
    Email: jakub-czerny@outlook.com
    Deep Learning for Computer Vision
    Python Version: 2.7
'''

# Nagivate to data/test/ and run command:
# find . -name \*.jpg -type f -delete

# The you're left with the folder structure and complete dataset in train/
# Now let's subselect __% of the data and move it to test by running this script


# This should be run from data/
import os, shutil
from random import shuffle

ratio = 0.3

path = os.getcwd()
folders = dict((x[0].split('/')[-1], []) for x in os.walk(path+'/train/') if x[0] != path+'/train/')
train = dict.fromkeys(folders)
test = dict.fromkeys(folders)

for f, _ in folders.iteritems():
    imgs = os.listdir(path+'/train/'+f)
    shuffle(imgs)
    idx = int(round(len(imgs)*ratio))
    train[f] = imgs[idx:]
    test[f] = imgs[:idx]

for f, imgs in test.iteritems():
    for img in imgs:
        src = path + '/train/' + f + '/' + img
        dst = path + '/test/'  + f + '/' + img
        shutil.move(src, dst)
