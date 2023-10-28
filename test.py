import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Выберите GPU (например, первый GPU)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')