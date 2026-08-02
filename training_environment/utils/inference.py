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
        def calc_length_penalty(seq_len):
            return ((5.0 + seq_len) ** len_penalty_alpha) / (6.0 ** len_penalty_alpha)

        completed_beams = []
        active_beams = [([BOS_ID], 0.0)]
        
        for _ in range(self.config["MAX_LENGTH"]):
            if not active_beams or len(completed_beams) >= beam_width:
                break
                
            # Gom tất cả active beams thành 1 batch duy nhất
            active_seqs = [b[0] for b in active_beams]
            output_tensor_batch = tf.constant(active_seqs, dtype=tf.int32)
            en_inp_batch = tf.tile(en_inp, [len(active_beams), 1])
            
            # Chạy model cho toàn bộ active beams
            predictions = self.model((en_inp_batch, output_tensor_batch), training=False)
            
            # Lấy predictions tại vị trí token cuối cùng: shape (num_active, vocab_size)
            last_token_preds = predictions[:, -1, :]
            log_probs_batch = tf.nn.log_softmax(last_token_preds).numpy()
            
            next_active_candidates = []
            
            for i, (seq, current_log_prob) in enumerate(active_beams):
                log_probs = log_probs_batch[i]
                top_indices = np.argsort(log_probs)[-beam_width:]
                
                for idx in top_indices:
                    idx = int(idx)
                    new_seq = seq + [idx]
                    new_log_prob = current_log_prob + log_probs[idx]
                    
                    if idx == EOS_ID:
                        seq_len = len(new_seq) - 1
                        score = new_log_prob / calc_length_penalty(seq_len)
                        completed_beams.append((score, new_seq))
                    else:
                        next_active_candidates.append((new_seq, new_log_prob))
            
            if not next_active_candidates:
                break

            next_active_candidates.sort(key=lambda x: x[1], reverse=True)
            active_beams = next_active_candidates[:beam_width]

        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed_beams[0][1]
        else:
            scored_active = []
            for seq, log_prob in active_beams:
                seq_len = len(seq) - 1
                score = log_prob / calc_length_penalty(seq_len)
                scored_active.append((score, seq))
            scored_active.sort(key=lambda x: x[0], reverse=True)
            best_seq = scored_active[0][1]
        
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