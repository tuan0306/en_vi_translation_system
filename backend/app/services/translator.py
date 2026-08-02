import os
import time
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from app.core.config import settings
from app.services.download import download_model_files

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

def log_softmax(x):
    x_max=np.max(x,axis=-1,keepdims=True)
    safe_x=x-x_max
    log_sum_exp = np.log(np.sum(np.exp(safe_x), axis=-1, keepdims=True))
    return safe_x-log_sum_exp

class TranslatorService:
    def __init__(self):
        self.session=None
        self.sp_en=None
        self.sp_vi=None
        self.max_length=settings.MAX_LENGTH

    def load_model(self):
        if self.session is not None:
            return
        
        paths=download_model_files()

        self.sp_en=spm.SentencePieceProcessor(model_file=paths["sp_en"])
        self.sp_vi=spm.SentencePieceProcessor(model_file=paths["sp_vi"])

        sess_options=ort.SessionOptions() # các luật lệ của ONNX Runtime
        sess_options.intra_op_num_threads=2 # số luồng trong 1 phép toán
        sess_options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL # xử lí tuần tự từng lớp

        self.session=ort.InferenceSession(
            paths["model"],
            sess_options,
            providers=["CPUExecutionProvider"] # sử dụng CPU
        )

        print("ONNX Model và Tokenizers được tải thành công!")

    
    def is_loaded(self):
        return self.session is not None and self.sp_en is not None and self.sp_vi is not None

    def beam_search_decode(self, encoder_input, beam_width=4, len_penalty_alpha=0.6):
        def calc_length_penalty(seq_len):
            return ((5.0 + seq_len) ** len_penalty_alpha) / (6.0 ** len_penalty_alpha)

        # lưu danh sách: (score, seq)
        completed_beams = []
        # lưu danh sách: (seq, log_prob)
        active_beams = [([BOS_ID], 0.0)]
        
        for _ in range(self.max_length):
            if not active_beams or len(completed_beams) >= beam_width:
                break
                
            # Gom tất cả active beams thành 1 batch 
            active_seqs = [b[0] for b in active_beams]
            decoder_input_batch = np.array(active_seqs, dtype=np.int32)
            encoder_input_batch = np.repeat(encoder_input, len(active_beams), axis=0)
            
            # Chạy ONNX session cho toàn bộ active beams
            outputs = self.session.run(
                None, 
                {
                    "encoder_input": encoder_input_batch,
                    "decoder_input": decoder_input_batch
                }
            )
            
            # Lấy dự đoán logits của token cuối cùng cho mỗi active beam: shape (num_active, vocab_size)
            last_token_logits = outputs[0][:, -1, :]
            log_probs_batch = log_softmax(last_token_logits)
            
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

            # Sắp xếp các active candidates theo log_prob để giữ lại top beam_width
            next_active_candidates.sort(key=lambda x: x[1], reverse=True)
            active_beams = next_active_candidates[:beam_width]

        # Lấy kết quả hoàn chỉnh tốt nhất
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

    def translate(self,text:str):
        if not self.is_loaded():
            raise RuntimeError("Mô hình chưa được tải")

        en_ids=[BOS_ID]+self.sp_en.encode(text,out_type=int)+[EOS_ID]
        encoder_input=np.array([en_ids],dtype=np.int32)

        vi_ids=self.beam_search_decode(encoder_input,beam_width=settings.BEAM_WIDTH,len_penalty_alpha=settings.LEN_PENALTY_ALPHA)

        translated_text=self.sp_vi.decode(vi_ids)
        return translated_text

translator_service = TranslatorService()