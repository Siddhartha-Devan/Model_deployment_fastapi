import tensorflow as tf

class IrisModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(IrisModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(4, activation = tf.keras.activations.relu)
        self.dense_2 = tf.keras.layers.Dense(3, activation = tf.keras.activations.relu)
        self.output_ = tf.keras.layers.Dense(num_classes, activation =  tf.keras.activations.softmax)
        

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        out = self.output_(x)
        return out
    




