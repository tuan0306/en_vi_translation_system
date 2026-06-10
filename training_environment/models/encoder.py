from models.attention import MultiHeadAttention
from models.layers import point_wise_feed_forward_network,PositionalEmbedding
import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super().__init__()
        self.mha=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.ffn=point_wise_feed_forward_network(d_model=d_model,dff=dff)
        
        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1=tf.keras.layers.Dropout(rate=rate)
        self.dropout2=tf.keras.layers.Dropout(rate=rate)
        
    def call(self, x, training, mask=None):
        attn_output,_=self.mha(x,x,x,mask)
        attn_output=self.dropout1(attn_output,training=training)
        out1=self.layernorm1(attn_output+x)
        
        ffn_output=self.ffn(out1)
        ffn_output=self.dropout2(ffn_output,training=training)
        out2 = self.layernorm2(ffn_output + out1)
        
        return out2
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,vocab_size,rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_embedding=PositionalEmbedding(vocab_size=vocab_size,d_model=d_model)
        
        self.enc_layers=[
            EncoderLayer(d_model=d_model,num_heads=num_heads,dff=dff,rate=rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self,x,training=False):
        padding_mask=self.pos_embedding.compute_mask(x)
        x=self.pos_embedding(x)
        x=self.dropout(x,training=training)
        
        for i in range(self.num_layers):
            x=self.enc_layers[i](x,training=training,mask=padding_mask)
        return x