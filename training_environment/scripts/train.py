import tensorflow as tf
from models.transformer import Transformer
from utils.optimizers import CustomSchedule, masked_loss, masked_accuracy
from utils.dataset_pipeline import TranslationDataset
import argparse
import yaml
import os
import json
import sentencepiece as spm
import pandas as pd

EN_SPM_PATH = 'data/tokenizer/spm_en.model'
VI_SPM_PATH = 'data/tokenizer/spm_vi.model'
TRAIN_TFRECORD = 'data/train.tfrecord'
VAL_TFRECORD = 'data/val.tfrecord'

TOTAL_TRAINING_SAMPLES = 389056

def smart_resume_training(checkpoint_path, latest_checkpoint_path, history_path):
    initial_epoch = 0
    best_val_loss = float('inf')
    checkpoint_to_load = None
    
    if os.path.exists(latest_checkpoint_path):
        checkpoint_to_load = latest_checkpoint_path
    elif os.path.exists(checkpoint_path):
        checkpoint_to_load = checkpoint_path
        
    if not os.path.exists(history_path) or checkpoint_to_load is None:
        print("Không tìm thấy lịch sử hoặc trọng số. Bắt đầu train mới từ Epoch 0.")
        return initial_epoch, best_val_loss, None
    
    try:
        df = pd.read_csv(history_path)
        if df.empty:
            print("File CSV rỗng. Bắt đầu train mới từ Epoch 0")
            return initial_epoch, best_val_loss, None
        
        last_row = df.iloc[-1]
        last_completed_epoch = int(last_row['epoch'])
        initial_epoch = last_completed_epoch + 1
        
        if 'val_loss' in df.columns:
            best_val_loss = df['val_loss'].min()
            
        print(f"Đã hoàn thành : {initial_epoch} Epochs")
        print(f"Kỷ lục Val Loss: {best_val_loss:.4f}")
        print(f"Sẵn sàng train tiếp từ Epoch thứ {initial_epoch + 1} sử dụng checkpoint: {checkpoint_to_load}")
        
        return initial_epoch, best_val_loss, checkpoint_to_load
    
    except Exception as e:
        print(f"Lỗi khi phục hồi quá trình train: {e}")
        print("Chuyển về chế độ train mới từ đầu")
        return 0, float('inf'), None
        

def main():
    with open('config.yaml','r',encoding='utf8') as f:
        config=yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers',type=int,default=config["NUM_LAYERS"])
    parser.add_argument('--d_model',type=int,default=config["D_MODEL"])
    parser.add_argument('--num_heads',type=int,default=config["NUM_HEADS"])
    parser.add_argument('--dff',type=int,default=config["DFF"])
    parser.add_argument('--max_length',type=int,default=config["MAX_LENGTH"])
    parser.add_argument('--batch_size',type=int,default=config["BATCH_SIZE"])
    parser.add_argument('--epochs',type=int,default=config["EPOCHS"])
    args=parser.parse_args()
    
    if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
    if not os.path.exists('report'):
        os.makedirs('report')

    checkpoint_filepath = 'checkpoints/transformer_best_model.weights.h5'
    latest_checkpoint_filepath = 'checkpoints/transformer_latest_model.weights.h5'
    history_filepath = 'report/training_history.csv'
    
    initial_epoch, best_val_loss, checkpoint_to_load = smart_resume_training(
        checkpoint_filepath, latest_checkpoint_filepath, history_filepath
    )
    
    dataset_builder=TranslationDataset(
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    train_ds=dataset_builder.create_dataset(tfrecord_file=TRAIN_TFRECORD,shuffle=True)
    val_ds=dataset_builder.create_dataset(tfrecord_file=VAL_TFRECORD)
    
    sp_en = spm.SentencePieceProcessor(model_file=EN_SPM_PATH)
    sp_vi = spm.SentencePieceProcessor(model_file=VI_SPM_PATH)
    INPUT_VOCAB_SIZE = sp_en.get_piece_size()
    TARGET_VOCAB_SIZE = sp_vi.get_piece_size()
    
    transformer=Transformer(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dff=args.dff,
        input_vocab_size=INPUT_VOCAB_SIZE,
        tgt_vocab_size=TARGET_VOCAB_SIZE
    )
    
    learning_rate=CustomSchedule(d_model=args.d_model)
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9, 
        beta_2=0.98, 
        epsilon=1e-9
    )
    
    transformer.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    for inp,tar in train_ds.take(1):
        _=transformer(inp)
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        transformer.load_weights(checkpoint_to_load)
        print(f"Đã load thành công trọng số từ: {checkpoint_to_load}")
        
    steps_per_epoch = TOTAL_TRAINING_SAMPLES // args.batch_size
    total_steps_run = initial_epoch * steps_per_epoch
    transformer.optimizer.iterations.assign(total_steps_run)
    print(f"Khởi tạo optimizer iterations ở step: {total_steps_run} (Epoch: {initial_epoch}, Steps/Epoch: {steps_per_epoch})")
        
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    # Checkpoint lưu weights có loss tốt nhất
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    model_checkpoint.best = best_val_loss
    
    # Checkpoint lưu weights mới nhất
    latest_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=latest_checkpoint_filepath,
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )
    
    logger = tf.keras.callbacks.CSVLogger(
        filename=history_filepath,
        separator=',',
        append=True
    )
    
    history = transformer.fit(
        train_ds,
        epochs=args.epochs+initial_epoch,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        callbacks=[early_stopping, model_checkpoint, latest_checkpoint, logger]
    )
    
        
if __name__=="__main__":
    main()