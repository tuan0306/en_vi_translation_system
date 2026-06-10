import tensorflow as tf

def scaled_dot_product_attention(q,k,v,mask=None):
    matmul_qk=tf.matmul(q,k,transpose_b=True)
    dk=tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits=matmul_qk/tf.math.sqrt(dk)
    
    if mask is not None:
        mask_float=tf.cast(mask,dtype=tf.float32)
        inverted_mask=1.0-mask_float
        if len(mask.shape) == 2:
            inverted_mask = inverted_mask[:, tf.newaxis, tf.newaxis, :]
            
        elif len(mask.shape) == 3:
            inverted_mask = inverted_mask[:, tf.newaxis, :, :]
        scaled_attention_logits+=(inverted_mask*-1e9)
    
    attention_weights=tf.nn.softmax(scaled_attention_logits,axis=-1)
    output=tf.matmul(attention_weights,v)
    return output,attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        
        assert d_model % self.num_heads == 0, "d_model phải chia hết cho num_heads"
        
        self.depth=d_model//num_heads
        
        self.wq=tf.keras.layers.Dense(d_model)
        self.wk=tf.keras.layers.Dense(d_model)
        self.wv=tf.keras.layers.Dense(d_model)
        
        self.dense=tf.keras.layers.Dense(d_model)
        
    def split_heads(self,x):
        x=tf.reshape(x,(tf.shape(x)[0],tf.shape(x)[1],self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3])
    
    def call(self,q,k,v,mask=None):
        q=self.wq(q)
        k=self.wk(k)
        v=self.wv(v)
        
        q=self.split_heads(q)
        k=self.split_heads(k)
        v=self.split_heads(v)
        
        scaled_attention,attention_weights=scaled_dot_product_attention(q,k,v,mask)
        scaled_attention=tf.transpose(scaled_attention,perm=[0,2,1,3])
        
        concat_attention=tf.reshape(scaled_attention,(tf.shape(q)[0],-1,self.d_model))
        
        output=self.dense(concat_attention)
        return output,attention_weights