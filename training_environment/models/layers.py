import tensorflow as tf
import numpy as np

def positional_encoding(length,depth):
    depth=depth/2
    positions=np.arange(length)[:,np.newaxis]
    depths=np.arange(depth)[np.newaxis,:]/depth
    angle_rates=1/(10000**depths)
    angle_rads=positions*angle_rates
    pos_encoding=np.concatenate([np.sin(angle_rads),np.cos(angle_rads)],axis=-1)
    return tf.cast(pos_encoding,dtype=tf.float32)[tf.newaxis,...]

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model,max_length=2048):
        super().__init__()
        self.d_model=d_model
        self.embedding=tf.keras.layers.Embedding(
            input_dim=vocab_size,output_dim=d_model,mask_zero=True
        )
        
        self.pos_encoding=positional_encoding(length=max_length,depth=d_model)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args,**kwargs)
    
    def call(self,x):
        length=tf.shape(x)[1]
        x=self.embedding(x)
        x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x=x+self.pos_encoding[:,:length,:]
        return x


def point_wise_feed_forward_network(d_model,dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    
def create_look_ahead_mask(size):
    mask=tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask