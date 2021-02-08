import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
from os import path
import time

THISPATH = path.dirname(__file__)
BATCH_SIZE = 64
EPOCHS = 50
PZ_DIM = 100
COLOR_DIMS = 1

class FaceDCGAN:
    def __init__(self, dims):
        self.dims = dims
        self.genm = self.__get_generator()
        self.dism = self.__get_discriminator()
        self.genm_opt = Adam(lr=0.0002, beta_1=0.5)
        self.dism_opt = Adam(lr=0.0002, beta_1=0.5)
        self.dis_loss = BinaryCrossentropy(from_logits=True)
        self.init = RandomNormal(stddev=0.2)
        self.check_seed = tf.random.normal([16, 100])
        self.dset_gen = ImageDataGenerator(
                preprocessing_function=lambda i: (i - 127.5) / 127.5
        )
        self.dataset = self.dset_gen.flow_from_directory(
                path.join(THISPATH, "faces"),
                target_size=(64, 64),
                color_mode="grayscale" if self.dims == 1 else "rgb",
                class_mode=None,
                batch_size=64
        )

    def __get_generator(self):
        m = tf.keras.Sequential()
        m.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(PZ_DIM,),
                         kernel_initializer=self.init))
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())

        m.add(layers.Reshape((4, 4, 1024)))
        assert m.output_shape == (None, 4, 4, 1024)

        m.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=self.init, use_bias=False))
        assert m.output_shape == (None, 8, 8, 512)
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())

        m.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=self.init, use_bias=False))
        assert m.output_shape == (None, 16, 16, 256)
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())

        m.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=self.init, use_bias=False))
        assert m.output_shape == (None, 32, 32, 128)
        m.add(layers.BatchNormalization())
        m.add(layers.ReLU())

        m.add(layers.Conv2DTranspose(self.dims, (5, 5), strides=(2, 2), padding='same',
                                   use_bias=False, activation='tanh',
                                   kernel_initializer=self.init))
        assert m.output_shape == (None, 64, 64, self.dims)

        return m

    def __get_discriminator(self):
        m = tf.keras.Sequential()
        m.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          kernel_initializer=self.init, input_shape=[64, 64, self.dims]))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(alpha=0.2))

        m.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                          kernel_initializer=self.init))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(alpha=0.2))

        m.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same',
                          kernel_initializer=self.init))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(alpha=0.2))

        m.add(layers.Flatten())
        m.add(layers.Dense(1, activation="sigmoid", kernel_initializer=self.init))

        return m

    def __save_epoch_image(self, epoch):
        preds = self.genm(self.check_seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            if self.dims == 1:
                plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            else:
                plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5)
            plt.axis("off")

        plt.savefig(path.join(THISPATH, "imgs", f"epoch_{epoch:04d}.png"))

    def __step(self, images):
        noise = tf.random.normal([BATCH_SIZE, PZ_DIM])

        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            fake_imgs = self.genm(noise, training=True)

            real_out = self.dism(images, training=True)
            fake_out = self.dism(fake_imgs, training=True)

            gen_loss = -tf.reduce_mean(tf.math.log(fake_out))
            dis_loss = self.dis_loss(tf.ones_like(real_out), real_out) + \
                    self.dis_loss(tf.zeros_like(fake_out), fake_out)

        gen_grad = gtape.gradient(gen_loss, self.genm.trainable_variables)
        dis_loss = dtape.gradient(dis_loss, self.dism.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grad, self.genm.trainable_variables))
        self.dis_opt.apply_gradients(zip(dis_loss, self.dism.trainable_variables))

    def __save_model(self):
        self.genm.save(path.join(THISPATH, "models", "generator"))
        self.dism.save(path.join(THISPATH, "models", "discriminator"))

    def train(self):
        for epoch in range(EPOCHS):
            start = time.time()

            steps_to_take = 70000 // BATCH_SIZE
            steps = 0
            for image_batch in self.dataset:
                self.__step(image_batch)
                steps += 1
                if steps == steps_to_take:
                    break

            self.__save_epoch_image(epoch)

            print(f"Epoch {epoch + 1:<4} | Time: {time.time() - start}")

        gifpath = path.join(THISPATH, "face_dcgan.gif")
        imgs_path = path.join(THISPATH, "imgs")
        with imageio.get_writer(gifpath, mode='I') as writer:
            files = sorted(glob.glob(path.join(imgs_path, "epoch*.png")))
            for filename in files:
                writer.append_data(imageio.imread(filename))
            writer.append_data(imageio.imread(filename))

        self.__save_model()
        print("Training Complete")

if __name__ == "__main__":
    gan = FaceDCGAN(COLOR_DIMS)
    gan.train()
