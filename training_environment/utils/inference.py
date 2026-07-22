import tensorflow as tf
import os
import yaml
import sentencepiece as spm
import sys
import numpy as np

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
    
    def beam_search_decode(self, en_inp, beam_width=4, len_penalty_alpha=0.6):
        beams = [([BOS_ID], 0.0, False)]
        
        for _ in range(self.config["MAX_LENGTH"]):
            candidates = []
            all_finished = True
            
            for seq, log_prob, finished in beams:
                if finished:
                    candidates.append((seq, log_prob, True))
                    continue
                
                all_finished = False
                
                output_tensor = tf.expand_dims(tf.constant(seq, dtype=tf.int32), 0)
                predictions = self.model((en_inp, output_tensor), training=False)
                predictions = predictions[0, -1, :]
                
                log_probs = tf.nn.log_softmax(predictions).numpy()
                
                top_indices = np.argsort(log_probs)[-beam_width:]
                for idx in top_indices:
                    idx = int(idx)
                    new_seq = seq + [idx]
                    new_log_prob = log_prob + log_probs[idx]
                    is_eos = (idx == EOS_ID)
                    candidates.append((new_seq, new_log_prob, is_eos))
                    
            if all_finished:
                break
                
            scored_candidates = []
            for seq, log_prob, finished in candidates:
                seq_len = len(seq) - 1
                penalty = ((5.0 + seq_len) ** len_penalty_alpha) / ((5.0 + 1.0) ** len_penalty_alpha)
                score = log_prob / penalty
                scored_candidates.append((score, seq, log_prob, finished))
                
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = []
            for score, seq, log_prob, finished in scored_candidates[:beam_width]:
                beams.append((seq, log_prob, finished))
                
        finished_beams = [b for b in beams if b[2] or b[0][-1] == EOS_ID]
        best_beam = finished_beams[0] if finished_beams else beams[0]
        
        best_seq = best_beam[0]
        result_ids = [t for t in best_seq if t not in [BOS_ID, EOS_ID]]
        return result_ids

    def translate(self, text):
        en_ids = [BOS_ID] + self.sp_en.encode(text, out_type=int) + [EOS_ID]
        en_inp = tf.expand_dims(tf.convert_to_tensor(en_ids, dtype=tf.int32), 0)
        
        vi_ids = self.beam_search_decode(en_inp, beam_width=4, len_penalty_alpha=0.6)
        translated_text = self.sp_vi.decode(vi_ids)
        return translated_text

translator=Translator(
    config_path='../config.yaml',
    en_tokenizer_path='../data/tokenizer/spm_en.model',
    vi_tokenizer_path = '../data/tokenizer/spm_vi.model',
    checkpoint_path='../checkpoints/transformer_best_model.weights.h5'
)    