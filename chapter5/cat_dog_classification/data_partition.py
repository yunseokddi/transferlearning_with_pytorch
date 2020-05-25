import glob
import numpy as np
import os
import shutil
from utils import log_progress

np.random.seed(42)

files = glob.glob('data/*')

cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]

#데이터 나누기
cat_train = np.random.choice(cat_files, size=10000, replace=False)
dog_train = np.random.choice(dog_files, size=10000, replace=False)
cat_files = list(set(cat_files) - set(cat_train))
dog_files = list(set(dog_files) - set(dog_train))


cat_test = np.random.choice(cat_files, size=2500, replace=False)
dog_test = np.random.choice(dog_files, size=2500, replace=False)

train_dir = 'data/train'
test_dir = 'data/test'

#데이터 합침
train_files = np.concatenate([cat_train, dog_train])
test_files = np.concatenate([cat_test, dog_test])

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

for fn in log_progress(train_files, name='Training Images'):
    shutil.copy(fn, train_dir)


for fn in log_progress(test_files, name='Test Images'):
    shutil.copy(fn, test_dir)

