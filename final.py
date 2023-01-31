import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import serial
import serial
import time
import datetime
import threading
import matplotlib.pyplot as plt

print(tf.version.VERSION)


IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 100
OUTPUT_CHANNELS = 3


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
       tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
         result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
        ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
        ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

  # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
      initializer = tf.random_normal_initializer(0., 0.02)

      inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
      tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

      x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

      down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
      down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
      down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

      zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
      conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

      batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

      leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

      zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

      last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

      return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
      real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

      generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

      total_disc_loss = real_loss + generated_loss

      return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './log'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    # display_list = [test_input[0], tar[0], prediction[0]]
    display_list = [ test_input[0], prediction[0]]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image']
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

TouchBoardData = []
countTimeEvent = threading.Event()

def create_scratch(df):

        maxDifference = 0.01
        #scratch = np.zeros(( 256, 512))
        scratch = np.zeros((256, 512, 3), dtype=np.uint8)
        scratch[0:256 , 0:512 ] = [255, 255, 255]

        dfX = df[df['Board'] == 'X']
        dfY = df[df['Board'] == 'Y']

        if len(dfX) == len(dfY):
                loopIndex = len(dfY)
        elif  len(dfX) > len(dfY):
                loopIndex = len(dfY)
        else: 
                loopIndex = len(dfX)

        electrodeX = 0
        electrodeY = 0

        for i in range(1, loopIndex):
                absDifference =  abs(float(dfX['Time'].iloc[i]) - float(dfY['Time'].iloc[i]))
                if absDifference > maxDifference:
                        continue
                if str(dfX['Value'].iloc[i]).count('1') == 1:
                        electrodeX = str(dfX['Value'].iloc[i]).find('1')
                if str(dfX['Value'].iloc[i]).count('1') == 1:
                        electrodeY = str(dfY['Value'].iloc[i]).find('1')
                if electrodeX > 0 and electrodeY > 0:
                        scratch[(electrodeY * 19 )][((electrodeX * 19) + 255)] =  [0, 0, 0]
                        electrodeX  = 0 
                        electrodeY = 0

        return scratch


def countTime():
        countTimeEvent.set()
        print('Event started')
        time.sleep(5)
        countTimeEvent.clear()
        print('Event finished')


def readSensor(usbPort, label, sensorData):
        countTimeEvent.wait()
        if countTimeEvent.is_set():
                readBool = True                
                try:
                        ser = serial.Serial(
                        port = usbPort, 
                        baudrate = 9600)
                except serial.SerialException:
                        print('Problem z portem' + label)
                        readBool = False
        while readBool:
                if countTimeEvent.is_set():
                        line = str(ser.readline(), 'utf-8')
                        print(line)
                        TouchedTime = str(time.time())
                        sensorData.append( [TouchedTime, line , label ])
                        
                        # print(TouchBoardData)
                else:
                        return None


def load_image_test(scrt):
    
    real_image = scrt[:, :256, :]
    input_image = scrt[:, 256:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

TouchBoardX = threading.Thread( name='AxisX',
                                target=readSensor,
                                args=('COM3', 'X', TouchBoardData ))

TouchBoardY = threading.Thread( name='AxisY',
                                target=readSensor,
                                args=('COM4', 'Y', TouchBoardData )) 

countTimeThread = threading.Thread( target=countTime )


countTimeThread.start()
TouchBoardX.start()
TouchBoardY.start()
countTimeThread.join()
TouchBoardX.join()
TouchBoardY.join()
df2 = pd.DataFrame(np.array(TouchBoardData) , columns=['Time', 'Value', 'Board' ])
scratch_dots = create_scratch(df2)


test_dataset = tf.data.Dataset.from_tensor_slices([scratch_dots])
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(5)

#print(test_dataset)

latest = tf.train.latest_checkpoint('log/')
checkpoint.restore(latest) 
print(latest)

for example_input, example_target in test_dataset:
        generate_images(generator, example_input, example_target)

