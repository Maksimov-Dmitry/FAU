import os.path
import json
from math import ceil
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import random
from os import listdir


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.files = listdir(self.file_path)
        if self.shuffle:
            random.shuffle(self.files)
        with open(label_path) as f:
            self.labels = json.load(f)
        self.epoch = 0
        self.batch_number = 0
        self.output = None

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        first_batch_index = self.batch_number*self.batch_size

        #due to the fact that the tests consider that if the last batch ends with the last file, then the epoch has not changed yet
        if first_batch_index == len(self.files):
            self.epoch += 1
            self.batch_number = 0
            if self.shuffle:
                random.shuffle(self.files)
    
        last_batch_index = (self.batch_number + 1)*self.batch_size
        self.batch_number += 1
        additional_files = 0
        is_changed_epoch = False
        if last_batch_index > len(self.files):
            is_changed_epoch = True
            additional_files = last_batch_index - len(self.files)
        images_names = self.files[first_batch_index:last_batch_index] + self.files[:additional_files]
    
        if is_changed_epoch:
            self.epoch += 1
            self.batch_number = 0
            if self.shuffle:
                random.shuffle(self.files)

        labels = np.array([self.labels[image_name.split('.')[0]] for image_name in images_names])
        images = np.array(
            [
                self.augment(np.load(os.path.join(self.file_path, image_name)))
                for image_name in images_names])
        self.output = (images.copy(), labels.copy())
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and random.choice([True, False]):
            img = np.flip(img, axis=1)
        if self.rotation and random.choice([True, False]):
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, k=angle // 90)
        
        img = resize(img, self.image_size)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        columns = 3
        rows = ceil(self.batch_size / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 6), dpi=90)
        axes_image = axes.flatten()[:self.batch_size]
        [ax.set_axis_off() for ax in axes.flatten()[self.batch_size:]]
        for i, ax in enumerate(axes_image):
            image = self.output[0][i]
            label = self.output[1][i]
            ax.imshow(image)
            ax.set_title(self.class_dict[label])
            ax.axis('off')
        plt.show()
        return
