import tensorflow as tf
import numpy as np
from imageio import imread, imsave
import glob
import cv2

images = glob.glob('gray/gray_val/*.jpg')
X_all = []
Y_all = []
WIDTH = 256
HEIGHT = 256
N = 10
images = np.random.choice(images, N, replace=False)
for image in images:
    img = cv2.imread(image)
    img = (img / 255. - 0.5) * 2
    # A2B
    X_all.append(img[:, :WIDTH, :])
    Y_all.append(img[:, WIDTH:, :])
X_all = np.array(X_all)
Y_all = np.array(Y_all)
print(X_all.shape, Y_all.shape)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('gray_model_ckpt/gray_model-100000.meta')
saver.restore(sess, 'gray_model_ckpt/gray_model-100000')
graph = tf.get_default_graph()
g = graph.get_tensor_by_name('generator/g:0')
X = graph.get_tensor_by_name('X:0')

gen_imgs = sess.run(g, feed_dict={X: X_all})
result = np.zeros([N * HEIGHT, WIDTH * 3, 3])

for i in range(N):
    result[i * HEIGHT: i * HEIGHT + HEIGHT, :WIDTH, :] = (X_all[i] + 1) / 2
    result[i * HEIGHT: i * HEIGHT + HEIGHT, WIDTH: 2 * WIDTH, :] = (Y_all[i] + 1) / 2
    result[i * HEIGHT: i * HEIGHT + HEIGHT, 2 * WIDTH:, :] = (gen_imgs[i] + 1) / 2
imsave('result.jpg', result)
