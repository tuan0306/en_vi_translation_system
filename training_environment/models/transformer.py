import tensorflow as tf
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,tgt_vocab_size,rate=0.1):
        super().__init__()
        
        self.encoder=Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            rate=rate
        )
        
        self.decoder=Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=tgt_vocab_size,
            rate=rate
        )
        
        self.output_bias=self.add_weight(
            name="output_bias",
            shape=(tgt_vocab_size,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self,inputs,training=False):
        inp,tar=inputs
        
        enc_padding_mask=self.encoder.pos_embedding.compute_mask(inp)
        enc_padding_mask = tf.cast(enc_padding_mask, tf.float32)
        
        enc_output=self.encoder(inp,training=training)
        
        dec_output=self.decoder(
            x=tar,enc_output=enc_output,enc_padding_mask=enc_padding_mask,training=training
        )

        embedding_matrix=self.decoder.pos_embedding.embedding.embeddings
        final_output=tf.matmul(dec_output,embedding_matrix,transpose_b=True)
        
        final_output=final_output+self.output_bias
        
        return final_output