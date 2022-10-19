from tensorflow import cast as tf_cast, float32 as tf_float32

def normalize(images, label):
    images = tf_cast(images, tf_float32)
    images /= 255
    return images, label

def arcface_format(image, label_group):
    return {'inp1': image, 'inp2': label_group}, label_group