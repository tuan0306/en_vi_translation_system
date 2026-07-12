import os
import argparse
import sentencepiece as spm

def train_spm(input_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        num_threads=os.cpu_count()
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size',type=int,default=16000)
    
    args=parser.parse_args()
    
    en_file='training_environment/data/raw/train.en'
    vi_file='training_environment/data/raw/train.vi'
    
    os.makedirs('training_environment/data/tokenizer',exist_ok=True)
    
    train_spm(en_file,'training_environment/data/tokenizer/spm_en',args.vocab_size)
    train_spm(vi_file,'training_environment/data/tokenizer/spm_vi',args.vocab_size)
    
if __name__ == "__main__":
    main()