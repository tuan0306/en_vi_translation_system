import tensorflow as tf
from models.transformer import Transformer
from utils.optimizers import CustomSchedule, masked_loss, masked_accuracy
from utils.dataset_pipeline import TranslationDataset
import argparse
import yaml
import os
import json
import sentencepiece as spm

EN_SPM_PATH = 'data/tokenizer/spm_en.model'
VI_SPM_PATH = 'data/tokenizer/spm_vi.model'
TRAIN_TFRECORD = 'data/train.tfrecord'
VAL_TFRECORD = 'data/val.tfrecord'

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
    
    dataset_builder=TranslationDataset(
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    train_ds=dataset_builder.create_dataset(tfrecord_file=TRAIN_TFRECORD)
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
    
    early_stopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('report'):
        os.makedirs('report')
        
        
    checkpoint_filepath = 'checkpoints/transformer_best_model.weights.h5'
    history_filepath = 'report/training_history.csv'
    
    if os.path.exists(checkpoint_filepath):
        for en_batch,vi_batch in train_ds.take(1):
            _=transformer(en_batch)
        transformer.load_weights(checkpoint_filepath)
    
        
    model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    
    logger=tf.keras.callbacks.CSVLogger(
        filename=history_filepath,
        separator=',',
        append=True
    )
    
    history=transformer.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint,logger]
    )
    
        
if __name__=="__main__":
    main()