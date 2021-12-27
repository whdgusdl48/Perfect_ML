import tensorflow as tf
import argparse
import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
# load Data

class Dataset():
    def __init__(self, data_folder, data_info, transform_target, transform_image):
        self.data_folder = data_folder
        self.data_info = data_info
        self.transform_target = transform_target
        self.transform_image = transform_image

        self.inputs = []
        self.outputs = []
        for idx, item in tqdm(data_info.iterrows(), desc='make dataset', total=len(data_info)):
            self.inputs.append(self.transform_image(Image.open(os.path.join(self.data_folder, item['input_0']))))
            self.outputs.append(self.transform_target(item['output_0']))


class Dataloader():
    def __init__(self, data_folder='./data (1)', batch_size=64,train_ratio = 0.9):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.train_folder = os.path.join(data_folder, 'train')
        self.test_folder = os.path.join(data_folder, 'test')

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        total_info = pd.read_csv(os.path.join(self.data_folder, 'train.csv'), sep='\t')
        self.train_info = total_info.sample(frac=self.train_ratio)
        self.val_info = total_info.drop(self.train_info.index)
        self.test_info = pd.read_csv(os.path.join(self.data_folder, 'test.csv'), sep='\t')

    def transform_target(self,labels):
        labels = tf.one_hot(labels,10)
        return labels

    def transform_image(self,images):
        images = np.array(images,dtype=np.float32)
        images = images / 255.0
        return images

    def setup(self, stage=None):
        self.prepare_data()
        transform_target = self.transform_target
        transform_image = self.transform_image
        self.train_dataset = Dataset(self.train_folder, self.train_info, transform_image=transform_image, transform_target=transform_target)
        self.val_dataset = Dataset(self.train_folder, self.val_info, transform_image=transform_image, transform_target=transform_target)
        self.test_dataset = Dataset(self.test_folder, self.test_info, transform_image=transform_image, transform_target=transform_target)

        train_ds = tf.data.Dataset.from_tensor_slices((self.train_dataset.inputs, self.train_dataset.outputs)).shuffle(10000).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((self.test_dataset.inputs, self.test_dataset.outputs)).batch(self.batch_size)
        self.train_dataset = train_ds
        self.test_dataset = test_ds


class Mnist_model():

    def __init__(self, DataLoader, epochs, input_shape, learning_rate):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.epochs = epochs
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.model()
        self.dataset = DataLoader
        self.dataset.setup()
        self.train_ds = self.dataset.train_dataset
        self.test_ds = self.dataset.test_dataset
        self.train_loss = tf.keras.metrics.Mean()
        self.train_acc = tf.keras.metrics.CategoricalAccuracy()
        self.test_loss = tf.keras.metrics.Mean()
        self.test_acc = tf.keras.metrics.CategoricalAccuracy()


    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            prediction = self.model(images)
            loss = self.loss(prediction, labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_acc(labels, prediction)

    def test_step(self, images, labels):

        with tf.GradientTape() as tape:
            prediction = self.model(images)

            loss = self.loss(prediction, labels)

        self.test_loss(loss)
        self.test_acc(labels, prediction)

    def train(self):

        for epoch in range(self.epochs):
            for images, labels in self.train_ds:
                self.train_step(images, labels)

            for images, labels in self.test_ds:
                self.test_step(images, labels)

            template = '에포크: {}, 손실: {:.5f}, 정확도: {:.2f}%,  테스트 손실: {:.5f}, 테스트 정확도: {:.2f}%'
            print(template.format(epoch + 1, self.train_loss.result(), self.train_acc.result() * 100,
                                  self.test_loss.result(), self.test_acc.result() * 100))

    def model(self):
        input = tf.keras.layers.Input(shape=self.input_shape, dtype=tf.float64)
        flatten = tf.keras.layers.Flatten()(input)
        layer_1 = tf.keras.layers.Dense(128)(flatten)
        activation = tf.keras.layers.ReLU()(layer_1)
        layer_2 = tf.keras.layers.Dense(256)(activation)
        activation_2 = tf.keras.layers.ReLU()(layer_2)
        layer_3 = tf.keras.layers.Dense(10)(activation_2)
        output = tf.keras.layers.Softmax()(layer_3)
        model = tf.keras.models.Model(input, output)
        return model


if __name__ == '__main__':
    data = Dataloader()
    model = Mnist_model(DataLoader=data,
                        epochs=10,
                        input_shape=(28, 28),
                        learning_rate=0.001)
    model.train()
