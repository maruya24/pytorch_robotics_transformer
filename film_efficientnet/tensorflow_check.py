import tensorflow as tf
from skimage import data
import numpy as np
import matplotlib.pyplot as plt

image_height = 512
image_width = 640

crop_size = 427

images = data.chelsea() # ndarray
images = tf.image.resize(images, (512, 640))
images = tf.cast(images, tf.uint8)
# images = tf.image.convert_image_dtype(images, tf.float32)

pad_then_crop = False
training = True


seed = tf.random.uniform(shape=(2,), maxval=2**30, dtype=tf.int32)
seed2 = tf.random.experimental.stateless_split(seed, num=1)[0]

if pad_then_crop:

    ud_pad = 40
    lr_pad = 100

    # 上下ともにun_pad=40だけpadding. 合計80
    # 左右ともにlr_pad=100だけpadding. 合計200
    images = tf.image.pad_to_bounding_box(
                        images,
                        offset_height=ud_pad,
                        offset_width=lr_pad,
                        target_height=image_height + 2 * ud_pad,
                        target_width=image_width + 2 * lr_pad)

    max_y = 2 * ud_pad
    max_x = 2 * lr_pad

    seed = tf.random.uniform(shape=(2,), maxval=2**30, dtype=tf.int32)
    seed2 = tf.random.experimental.stateless_split(seed, num=1)[0]

    offset_y = tf.random.stateless_uniform((),
                                    maxval=max_y + 1,
                                    dtype=tf.int32,
                                    seed=seed)
    offset_x = tf.random.stateless_uniform((),
                                    maxval=max_x + 1,
                                    dtype=tf.int32,
                                    seed=seed2)
    images = tf.image.crop_to_bounding_box(images, offset_y, offset_x,
                                                image_height, image_width)

else:
    # Standard cropping.
    max_y = image_height - crop_size
    max_x = image_width - crop_size

    if training:
      offset_y = tf.random.stateless_uniform((),
                                             maxval=max_y + 1,
                                             dtype=tf.int32,
                                             seed=seed)
      offset_x = tf.random.stateless_uniform((),
                                             maxval=max_x + 1,
                                             dtype=tf.int32,
                                             seed=seed2)
      images = tf.image.crop_to_bounding_box(images, offset_y, offset_x,
                                             crop_size, crop_size)
    else:
      images = tf.image.crop_to_bounding_box(images, max_y // 2, max_x // 2,
                                             crop_size, crop_size)

image_show = images.numpy()
# image_show = np.clip(image_show, 0, 1)
plt.imshow(image_show)
plt.show()
