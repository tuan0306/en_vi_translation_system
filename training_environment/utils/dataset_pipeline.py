import tensorflow as tf
import sentencepiece as spm

class TranslationDataset:
    def __init__(self,spm_en_path,spm_vi_path,max_length=64,batch_size=64):
        self.sp_en=spm.SentencePieceProcessor(model_file=spm_en_path)
        self.sp_vi=spm.SentencePieceProcessor(model_file=spm_vi_path)
        self.max_length=max_length
        self.batch_size=batch_size
    
    def encode(self,en_text,vi_text):
        en_str=en_text.numpy().decode('utf-8')
        vi_str=vi_text.numpy().decode('utf-8')
        
        en_tokens=[2]+self.sp_en.encode_as_ids(en_str)+[3]
        vi_tokens=[2]+self.sp_vi.encode_as_ids(vi_str)+[3]
        
        return en_tokens,vi_tokens
    
    def tf_encode(self,en_text,vi_text):
        en_enc,vi_enc=tf.py_function(
            self.encode,
            inp=[en_text,vi_text],
            Tout=[tf.int32,tf.int32]
        )
        
        en_enc.set_shape([None])
        vi_enc.set_shape([None])
        
        return en_enc,vi_enc
    
    def filter_length(self, en, vi):
        return tf.logical_and(tf.size(en)<=self.max_length,tf.size(vi)<=self.max_length)
    
    def create_dataset(self,en_file,vi_file):
        dataset_en=tf.data.TextLineDataset(en_file)
        dataset_vi=tf.data.TextLineDataset(vi_file)
        
        dataset=tf.data.Dataset.zip((dataset_en,dataset_vi))
        
        dataset=dataset.map(self.tf_encode,num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset=dataset.filter(self.filter_length)
        
        boundaries=[10,20,30,40]
        batch_sizes=[self.batch_size]*(len(boundaries)+1)
        
        dataset=dataset.bucket_by_sequence_length(
            element_length_func=lambda en,vi: tf.maximum(tf.size(en),tf.size(vi)),
            bucket_boundaries=boundaries,
            bucket_batch_sizes=batch_sizes,
            padded_shapes=([None],[None]),
            padding_values=(0,0)
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)