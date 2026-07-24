import os
import yaml
import sys
import tensorflow as tf
import shutil
from models.transformer import Transformer
import sentencepiece as spm

EN_SPM_PATH = 'data/tokenizer/spm_en.model'
VI_SPM_PATH = 'data/tokenizer/spm_vi.model'

def export_model_to_onnx():
    with open('config.yaml','r',encoding='utf-8') as f:
        config=yaml.safe_load(f)

    sp_en = spm.SentencePieceProcessor(model_file=EN_SPM_PATH)
    sp_vi = spm.SentencePieceProcessor(model_file=VI_SPM_PATH)
    INPUT_VOCAB_SIZE = sp_en.get_piece_size()
    TARGET_VOCAB_SIZE = sp_vi.get_piece_size()

    model=Transformer(
        num_layers=config["NUM_LAYERS"],
        num_heads=config["NUM_HEADS"],
        d_model=config["D_MODEL"],
        dff=config["DFF"],
        input_vocab_size=INPUT_VOCAB_SIZE,
        tgt_vocab_size=TARGET_VOCAB_SIZE
    )

    max_length=config["MAX_LENGTH"]
    dummy_en=tf.zeros((1,max_length),dtype=tf.int32)
    dummy_vi=tf.zeros((1,max_length-1),dtype=tf.int32)

    _=model((dummy_en,dummy_vi),training=False)

    weights_path = 'checkpoints/transformer_best_model.weights.h5'
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Đã load trọng số từ: {weights_path}")
    else:
        print("Lỗi: Không tìm thấy file trọng số.")
        return

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name="encoder_input"),
    #     tf.TensorSpec(shape=[None, max_length - 1], dtype=tf.int32, name="decoder_input")
    # ])
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="encoder_input"),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="decoder_input")
    ])
    def serving_fn(encoder_input,encoder_output):
        return model((encoder_input,encoder_output),training=False)

    saved_model_dir=r"C:\Temp\my_model_export"
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    tf.saved_model.save(model,saved_model_dir,signatures={"serving_default":serving_fn})
    print(f"Đã xuất ra định dạng TensorFlow SavedModel tại thư mục: {saved_model_dir}")

if __name__ == "__main__":
    export_model_to_onnx()
    