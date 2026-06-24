import tensorflow as tf
import sentencepiece as spm

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tf_record(en_path, vi_path, output_path, sp_en_path, sp_vi_path):
    sp_en=spm.SentencePieceProcessor(model_file=sp_en_path)
    sp_vi=spm.SentencePieceProcessor(model_file=sp_vi_path)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        with open(en_path, 'r', encoding='utf-8') as f_en, \
             open(vi_path, 'r', encoding='utf-8') as f_vi:
                 for i,(line_en,line_vi) in enumerate(zip(f_en,f_vi)):
                     en_ids=sp_en.encode(line_en.strip(),out_type=int)
                     vi_ids=sp_vi.encode(line_vi.strip(),out_type=int)
                     
                     feature={
                         'en':int64_feature(en_ids),
                         'vi':int64_feature(vi_ids)
                     }
                     
                     features=tf.train.Features(feature=feature)
                     example=tf.train.Example(features=features)
                     writer.write(example.SerializeToString())
                     
                     if (i + 1) % 50000 == 0:
                        print(f"Đang xử lý dòng {i + 1}...")
                     
def main():
    EN_MODEL='training_environment/data/tokenizer/spm_en.model'
    VI_MODEL='training_environment/data/tokenizer/spm_vi.model'
    
    EN_TRAIN='training_environment/data/raw/train.en'
    VI_TRAIN='training_environment/data/raw/train.vi'
    OUT_TRAIN='training_environment/data/train.tfrecord'
    
    EN_VAL='training_environment/data/raw/val.en'
    VI_VAL='training_environment/data/raw/val.vi'
    OUT_VAL='training_environment/data/val.tfrecord'
    
    EN_TEST='training_environment/data/raw/test.en'
    VI_TEST='training_environment/data/raw/test.vi'
    OUT_TEST='training_environment/data/test.tfrecord'
    
    create_tf_record(
        en_path=EN_TRAIN, 
        vi_path=VI_TRAIN, 
        output_path=OUT_TRAIN, 
        sp_en_path=EN_MODEL, 
        sp_vi_path=VI_MODEL
        )
    
    create_tf_record(
        en_path=EN_VAL, 
        vi_path=VI_VAL, 
        output_path=OUT_VAL, 
        sp_en_path=EN_MODEL, 
        sp_vi_path=VI_MODEL
        )
    
    create_tf_record(
        en_path=EN_TEST, 
        vi_path=VI_TEST, 
        output_path=OUT_TEST, 
        sp_en_path=EN_MODEL, 
        sp_vi_path=VI_MODEL
        )
    
if __name__=="__main__":
    main()