import pathlib
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import tarfile
import tempfile

# https://www.tensorflow.org/datasets/catalog/overview
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
temp_dir = tempfile.mkdtemp()
archive = tf.keras.utils.get_file(
    origin=dataset_url,
    extract=False,
    cache_dir=temp_dir
)

with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(path='./media')
exit()





# archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
# data_dir = pathlib.Path(archive).with_suffix('')
# with tarfile.open(archive, 'r:gz') as tar:
#     tar.extractall(path='./media')  

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))
# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[1]))

# batch_size = 32
# img_height = 180
# img_width = 180

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# # train
# batch_size = 32
# img_height = 180
# img_width = 180

# class_names = train_ds.class_names
# print(class_names)


# flowers_photos/
#   daisy/
#   dandelion/
#   roses/
#   sunflowers/
#   tulips/