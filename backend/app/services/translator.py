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

        sess_options=ort.SessionOptions() # các luật lệ của ONNX Rentime
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
        # Mỗi phần tử trong beams gồm: (danh_sách_tokens, tổng_log_prob, trạng_thái_kết_thúc)
        beams = [([BOS_ID], 0.0, False)]
        
        for _ in range(self.max_length):
            candidates = []
            all_finished = True
            
            for seq, log_prob, finished in beams:
                if finished:
                    candidates.append((seq, log_prob, True))
                    continue
                
                all_finished = False
                
                decoder_input = np.array([seq], dtype=np.int32)
                
                # Chạy mô hình qua ONNX session
                outputs = self.session.run(
                    None, 
                    {
                        "encoder_input": encoder_input,
                        "decoder_input": decoder_input
                    }
                )
                
                # Lấy dự đoán logits của token cuối cùng trong chuỗi hiện tại
                predictions = outputs[0][0, len(seq) - 1, :]
                
                # Áp dụng Log Softmax
                log_probs = log_softmax(predictions)
                
                # Lấy ra beam_width chỉ số tốt nhất
                top_indices = np.argsort(log_probs)[-beam_width:]
                for idx in top_indices:
                    idx = int(idx)
                    new_seq = seq + [idx]
                    new_log_prob = log_prob + log_probs[idx]
                    is_eos = (idx == EOS_ID)
                    candidates.append((new_seq, new_log_prob, is_eos))
                    
            if all_finished:
                break
                
            # Áp dụng Length Penalty
            scored_candidates = []
            for seq, log_prob, finished in candidates:
                seq_len = len(seq) - 1
                penalty = ((5.0 + seq_len) ** len_penalty_alpha) / ((5.0 + 1.0) ** len_penalty_alpha)
                score = log_prob / penalty
                scored_candidates.append((score, seq, log_prob, finished))
                
            # Sắp xếp các ứng cử viên theo điểm số từ cao xuống thấp
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Giữ lại beam_width kết quả tốt nhất
            beams = []
            for score, seq, log_prob, finished in scored_candidates[:beam_width]:
                beams.append((seq, log_prob, finished))
                
        # Lọc ra các beams đã gặp EOS_ID hoặc lấy beam tốt nhất
        finished_beams = [b for b in beams if b[2] or b[0][-1] == EOS_ID]
        best_beam = finished_beams[0] if finished_beams else beams[0]
        
        best_seq = best_beam[0]
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