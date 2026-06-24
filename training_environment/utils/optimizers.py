import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model,warmup_steps=4000):
        super().__init__()
        self.d_model=d_model
        self.d_model_cast=tf.cast(self.d_model,tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Công thức: (d_model^-0.5) * min(step^-0.5, step * warmup_steps^-1.5)
        arg1=tf.math.rsqrt(step)
        arg2=step*(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model_cast)*tf.math.minimum(arg1,arg2)

def masked_loss(real, pred):
    real = tf.cast(real, dtype=tf.int32)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def masked_accuracy(real, pred):
    real = tf.cast(real, dtype=tf.int32)
    pred_classes = tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)
    accuracies = tf.equal(real, pred_classes)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)