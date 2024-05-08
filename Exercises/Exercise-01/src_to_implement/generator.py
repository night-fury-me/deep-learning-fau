import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

class ImageTransformation:
    def apply(self, image):
        pass

class RotateTransformation(ImageTransformation):
    def __init__(self, enabled = False):
        self.enabled = enabled

    def apply(self, image):
        if not self.enabled:
            return image
            
        rotation_count = np.random.choice((1, 2, 3))  # 1-> 90, 2->180, 3->270

        # print(f"RotateTransformation applied!")
        return np.rot90(image, rotation_count)

class MirrorTransformation(ImageTransformation):
    def __init__(self, enabled = False):
        self.enabled = enabled
        
    def apply(self, image):
        if not self.enabled:
            return image

        # Random Vertical Flip
        if np.random.choice((True, False)):
            image = np.flip(image, axis = 0)

        # Random Horizontal Flip
        if np.random.choice((True, False)):
            image = np.flip(image, axis = 1)

        # print(f"MirrorTransformation applied!")
        return image

class ResizeTransformation(ImageTransformation):
    def __init__(self, expected_image_shape):
        self.expected_image_shape = expected_image_shape
        
    def apply(self, image):
        if image.shape == self.expected_image_shape:
            return image
        # print(f"ResizeTransformation applied!")
        return np.resize(image, self.expected_image_shape)

class Image(np.ndarray):
    def __new__(cls, image_array):
        image = np.asarray(image_array).view(cls)
        return image

    def apply(self, *transformations):
        for transformation in transformations:
            self = transformation.apply(self)

        # print(f"Called!")
        return self


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
        
        # TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(self.label_path, 'r') as f:
            self.image_index_to_label = json.load(f)

        self.__dataset_size = len(self.image_index_to_label)
        self.__data_index = np.arange(self.__dataset_size)

        if self.shuffle:
            np.random.shuffle(self.__data_index)

        self.__epoch_number = 0
        self.__batch_number = 0

        if self.__dataset_size < self.batch_size:
            self.batch_size = self.__dataset_size
        
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        
        start_index = self.__batch_number * self.batch_size

        if start_index >= self.__dataset_size:
            self.__batch_number = 0
            self.__epoch_number += 1
            if self.shuffle:
                np.random.shuffle(self.__data_index)
        
        padded_element_count = max(0, start_index + self.batch_size - self.__dataset_size)
        
        current_batch_indexs = np.concatenate((
            self.__data_index[start_index: start_index + self.batch_size],
            self.__data_index[:padded_element_count]
        ))
        
        if self.shuffle:
            np.random.shuffle(current_batch_indexs)

        images, labels = list(), list()

        for index in current_batch_indexs:
            image = self.augment(Image(np.load(f"{self.file_path}/{index}.npy")))
            label = self.image_index_to_label[str(index)]

            images.append(image)
            labels.append(label)

        batched_image = np.stack(images, axis = 0)
        batched_label = np.stack(labels, axis = 0)

        self.__batch_number += 1

        return batched_image, batched_label

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        return img.apply(
            ResizeTransformation(expected_image_shape = self.image_size),
            RotateTransformation(enabled = self.rotation),
            MirrorTransformation(enabled = self.mirroring)
        )

    def current_epoch(self):
        # return the current epoch number
        return self.__epoch_number

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]
        
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        
        images, labels = self.next()
        
        columns = 3
        rows = self.batch_size // 3
        
        if self.batch_size % 3 != 0:
            rows += 1
        
        fig  = plt.figure(figsize = (10,10))
        
        for i, (image, label) in enumerate(zip(images, labels)):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(image.astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(self.class_name(label))
            
        plt.show()