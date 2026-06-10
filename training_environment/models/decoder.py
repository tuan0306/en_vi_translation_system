import tensorflow as tf
from models.attention import MultiHeadAttention
from models.layers import point_wise_feed_forward_network,PositionalEmbedding, create_look_ahead_mask

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super().__init__()
        self.mha1=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.mha2=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        
        self.ffn=point_wise_feed_forward_network(d_model=d_model,dff=dff)
        
        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1=tf.keras.layers.Dropout(rate=rate)
        self.dropout2=tf.keras.layers.Dropout(rate=rate)
        self.dropout3=tf.keras.layers.Dropout(rate=rate)
        
    def call(self, x, enc_output, training, decoder_self_attention_mask, padding_mask):
        attn1, attn_weights_block1=self.mha1(x,x,x,mask=decoder_self_attention_mask)
        attn1=self.dropout1(attn1,training=training)
        out1=self.layernorm1(attn1+x)
        
        attn2, attn_weights_block2=self.mha2(out1,enc_output,enc_output,mask=padding_mask)
        attn2=self.dropout2(attn2,training=training)
        out2=self.layernorm2(attn2+out1)
        
        ffn_output=self.ffn(out2)
        ffn_output=self.dropout3(ffn_output,training=training)
        out3=self.layernorm3(ffn_output+out2)
        
        return out3,attn_weights_block1,attn_weights_block2
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate=0.1):
        super().__init__()
        self.d_model=d_model
        self.num_layers=num_layers
        
        self.pos_embedding=PositionalEmbedding(vocab_size=vocab_size,d_model=d_model)
        
        self.dec_layers=([
            DecoderLayer(d_model=d_model,num_heads=num_heads,dff=dff,rate=rate)
            for _ in range(num_layers)
        ])
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output,enc_padding_mask, training=False):
        tgt_padding_mask=self.pos_embedding.compute_mask(x)
        tgt_padding_mask=tf.cast(tgt_padding_mask,tf.float32)
        
        tgt_seq_length=tf.shape(x)[1]
        look_ahead=create_look_ahead_mask(tgt_seq_length)
        
        tgt_padding_mask_expanded=tgt_padding_mask[:,tf.newaxis,:]
        
        decoder_self_attention_mask = tf.minimum(look_ahead, tgt_padding_mask_expanded)
        
        x=self.pos_embedding(x)
        x=self.dropout(x,training=training)
        
        for i in range(self.num_layers):
            x,block1,block2=self.dec_layers[i](
                x,enc_output,training=training,
                decoder_self_attention_mask=decoder_self_attention_mask,
                padding_mask=enc_padding_mask
            )
        return x