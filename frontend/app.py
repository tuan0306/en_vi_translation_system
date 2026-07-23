import streamlit as st
import streamlit.components.v1 as components
import time
from api_client import api_client

# 1. Cấu hình trang (Page Configuration)
st.set_page_config(
    page_title="Hệ Thống Dịch Máy Anh - Việt (EN-VI)",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS để trang trí giao diện và hộp kết quả dịch thuật
st.markdown("""
<style>
    /* Ép buộc ô nhập liệu (st.text_area) có nền tối và chữ sáng rõ nét */
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #ffffff !important; /* Đổi sang màu trắng để nhìn rõ hơn */
        font-size: 1.1rem !important;
        border: 1px solid #30363d !important;
    }
    
    /* Tăng độ sáng và độ rõ cho chữ ẩn gợi ý (placeholder) */
    .stTextArea textarea::placeholder {
        color: #8b949e !important;
        opacity: 1 !important;
    }
    
    /* Màu chữ hiển thị đếm số ký tự ở góc ô nhập liệu */
    .stTextArea div {
        color: #8b949e !important;
    }

    /* Hộp kết quả dịch mượt mà */
    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 16px;
        border-radius: 8px;
        min-height: 180px;
        font-size: 1.1rem;
        white-space: pre-wrap;
        color: #f0f6fc; /* Màu chữ trắng sáng */
    }
</style>
""", unsafe_allow_html=True)

# 3. Khởi tạo State lưu trữ lịch sử dịch nếu chưa có
if "history" not in st.session_state:
    st.session_state.history = []

# 4. Giao diện Sidebar (Thanh bên)
with st.sidebar:
    # Sử dụng biểu tượng Unicode quả địa cầu thay cho ảnh ngoài để tránh lỗi hiển thị
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>🌐</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0; margin-bottom: 20px;'>EN-VI Translator</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Kiểm tra trạng thái Backend
    st.subheader("Trạng Thái Hệ Thống")
    health = api_client.check_health()
    
    if health["status"] == "healthy" and health["model_loaded"]:
        st.success("🟢 Online (Mô hình đã tải)")
    elif health["status"] == "healthy" and not health["model_loaded"]:
        st.warning("🟡 Online (Đang tải mô hình...)")
    elif health["status"] == "offline":
        st.error("🔴 Offline (Chưa bật Backend)")
    else:
        st.error("🔴 Lỗi kết nối")

    st.markdown(f"**Thiết bị chạy:** {health.get('device', 'Không xác định')}")
    st.markdown("---")

    # Hiển thị Lịch sử Dịch
    st.subheader("📜 Lịch Sử Lượt Dịch")
    if not st.session_state.history:
        st.info("Chưa có lượt dịch nào.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{idx+1}. {item['source'][:20]}...", expanded=False):
                st.markdown(f"**Gốc:** *{item['source']}*")
                st.markdown(f"**Dịch:** **{item['translated']}**")
                st.caption(f"Thời gian: {item['latency']}s | Cached: {item['cached']}")
        
        if st.button("Xóa Lịch Sử"):
            st.session_state.history = []
            st.rerun()

# 5. Giao diện Chính (Main UI)
st.title("🤖 Trình Dịch Máy Anh - Việt")
st.markdown("Ứng dụng dịch máy sử dụng mô hình **Transformer** tự code kết hợp công nghệ tối ưu hóa **ONNX Runtime** & **Beam Search**.")

# Khung chứa chính (Double-Pane layout)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🇺🇸 Tiếng Anh (English)")
    
    source_text = st.text_area(
        label="Nhập văn bản tiếng Anh cần dịch:",
        placeholder="Type here to translate...",
        height=180,
        label_visibility="collapsed"
    )
    st.caption("💡 Mẹo: Nhấp chuột vào ô trên và nhấn phím **Ctrl + V** để dán nhanh văn bản.")
    
    # Nút bấm trigger dịch thuật
    btn_translate = st.button("Dịch sang Tiếng Việt ➡️", use_container_width=True)

# Khởi tạo các giá trị trống cho kết quả
translated_text = ""
latency = 0.0
is_cached = False

with col2:
    st.markdown("### 🇻🇳 Tiếng Việt (Vietnamese)")
    
    # Khi người dùng bấm nút dịch hoặc gõ Enter
    if btn_translate and source_text.strip():
        if health["status"] == "offline":
            st.error("Không thể dịch. Vui lòng khởi động Backend API trước (cổng 8000).")
        else:
            with st.spinner("Đang xử lý dịch máy..."):
                # Gọi API dịch thuật từ client
                result = api_client.translate(source_text)
                
                if result["success"]:
                    translated_text = result["translated_text"]
                    latency = result["latency_seconds"]
                    is_cached = result["cached"]
                    
                    # Lưu vào lịch sử dịch
                    st.session_state.history.append({
                        "source": source_text.strip(),
                        "translated": translated_text,
                        "latency": latency,
                        "cached": "Có" if is_cached else "Không"
                    })
                else:
                    st.error(f"Lỗi: {result['error']}")
    
    # Hiển thị box kết quả dịch
    if translated_text:
        st.markdown(f'<div class="result-box">{translated_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box" style="color: #8b949e;">Bản dịch sẽ xuất hiện ở đây...</div>', unsafe_allow_html=True)

    # Hiển thị các thông số thời gian
    if translated_text:
        st.caption(f"⏱️ Thời gian xử lý: **{latency} giây** | 💾 Lấy từ Cache: **{'Có' if is_cached else 'Không'}**")
