import tensorflow as tf
import os
import yaml
import sentencepiece as spm
import sys

sys.path.append(os.path.abspath('training_environment'))
from models.transformer import Transformer

PAD_ID=0
BOS_ID=2
EOS_ID=3

class Translator():
    def __init__(self,config_path,en_tokenizer_path,vi_tokenizer_path,checkpoint_path):
        with open(config_path,'r',encoding='utf-8') as f:
            self.config=yaml.safe_load(f)

        self.sp_en=spm.SentencePieceProcessor(model_file=en_tokenizer_path)
        self.sp_vi=spm.SentencePieceProcessor(model_file=vi_tokenizer_path)

        self.model=Transformer(
            num_layers=self.config["NUM_LAYERS"],
            num_heads=self.config["NUM_HEADS"],
            d_model=self.config["D_MODEL"],
            dff=self.config["DFF"],
            input_vocab_size=self.sp_en.get_piece_size(),
            tgt_vocab_size=self.sp_vi.get_piece_size()
        )

        dummy_en=tf.zeros((1,self.config["MAX_LENGTH"]),dtype=tf.int32)
        dummy_vi_inp=tf.zeros((1,self.config["MAX_LENGTH"]-1),dtype=tf.int32)
        _=self.model((dummy_en,dummy_vi_inp),training=False)
        
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print(f"Tải thành công trọng số tại: {checkpoint_path}")
        else:
            print(f"Không tìm thấy file trọng số tại '{checkpoint_path}'!")
    
    def translate(self,text):
        en_ids=[BOS_ID]+self.sp_en.encode(text,out_type=int)+[EOS_ID]
        en_inp=tf.expand_dims(tf.convert_to_tensor(en_ids,dtype=tf.int32),0)

        decoder_inp=[BOS_ID]
        output=tf.expand_dims(decoder_inp,0)

        for _ in range(self.config["MAX_LENGTH"]):
            predictions=self.model((en_inp,output),training=False)
            predictions=predictions[:,-1:,:]
            predicted_id=tf.cast(tf.argmax(predictions,axis=-1),tf.int32)

            if predicted_id[0,0]==EOS_ID:
                break

            output=tf.concat([output,predicted_id],axis=-1)

        raw_ids = tf.squeeze(output, axis=0)[1:].numpy().tolist()
        vi_ids = [t for t in raw_ids if t not in [PAD_ID, BOS_ID, EOS_ID]]

        translated_text=self.sp_vi.decode(vi_ids)
        return translated_text

translator=Translator(
    config_path='../config.yaml',
    en_tokenizer_path='../data/tokenizer/spm_en.model',
    vi_tokenizer_path = '../data/tokenizer/spm_vi.model',
    checkpoint_path='../checkpoints/transformer_best_model.weights.h5'
)    