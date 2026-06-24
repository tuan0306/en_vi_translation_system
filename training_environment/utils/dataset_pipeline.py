import tensorflow as tf
import sentencepiece as spm

class TranslationDataset:
    def __init__(self,max_length=64,batch_size=64):
        self.max_length=max_length
        self.batch_size=batch_size
        self.START = tf.constant([2], dtype=tf.int32)
        self.END = tf.constant([3], dtype=tf.int32)
    
    def parse_tfrecord(self, serialized_example):
        feature_description={
            'en':tf.io.VarLenFeature(tf.int64),
            'vi':tf.io.VarLenFeature(tf.int64)
        }
        
        example=tf.io.parse_single_example(serialized_example,feature_description)
        en=tf.sparse.to_dense(example['en'])
        vi=tf.sparse.to_dense(example['vi'])
        
        en=tf.cast(en,tf.int32)
        vi=tf.cast(vi,tf.int32)
        
        en = tf.concat([self.START, en, self.END], axis=-1)
        vi = tf.concat([self.START, vi, self.END], axis=-1)
        
        return en, vi
    
    def filter_length(self, en, vi):
        return tf.logical_and(tf.size(en)<=self.max_length,tf.size(vi)<=self.max_length)
    
    def split_target(self, en, vi):
        vi_inp = vi[:, :-1]
        vi_real = vi[:, 1:]
        return (en, vi_inp), vi_real
    
    def create_dataset(self,tfrecord_file):
        dataset=tf.data.TFRecordDataset(tfrecord_file)
        
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset=dataset.filter(self.filter_length)

        dataset=dataset.shuffle(buffer_size=20000)
        
        boundaries=[10,20,30,40]
        batch_sizes=[self.batch_size]*(len(boundaries)+1)
        
        dataset=dataset.bucket_by_sequence_length(
            element_length_func=lambda en,vi: tf.maximum(tf.size(en),tf.size(vi)),
            bucket_boundaries=boundaries,
            bucket_batch_sizes=batch_sizes,
            #padded_shapes=([None],[None]),
            padded_shapes=([self.max_length],[self.max_length]),
            padding_values=(0,0)
        )

        dataset = dataset.map(self.split_target, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset.prefetch(tf.data.AUTOTUNE)