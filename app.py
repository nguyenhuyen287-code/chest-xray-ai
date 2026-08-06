import streamlit as st
import cv2
import numpy as np
import pydicom
from PIL import Image
import time
import pandas as pd

# ==========================================
# 1. CẤU HÌNH TRANG WEB (CONFIG)
# ==========================================
st.set_page_config(
    page_title="Hệ thống AI Phân Tích X-Quang Phổi",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện chuẩn y tế
st.markdown("""
    <style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #003366;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #004080;
        font-size: 18px;
        font-weight: bold;
    }
    .stAlert {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM TIỀN XỬ LÝ ẢNH Y TẾ & GIẢ LẬP AI
# ==========================================
def process_dicom_or_image(file):
    """Tiền xử lý ảnh DICOM hoặc PNG/JPG"""
    file_bytes = file.read()
    filename = file.name.lower()
    
    if filename.endswith('.dcm'):
        # Đọc file DICOM
        dicom = pydicom.dcmread(pydicom.filelike.BytesIO(file_bytes))
        img = dicom.pixel_array.astype(float)
        
        # Áp dụng Cửa sổ phổi (Lung Windowing)
        wc, ww = -600, 1200
        img_min = wc - ww // 2
        img_max = wc + ww // 2
        img = np.clip(img, img_min, img_max)
        img = ((img - img_min) / ww) * 255.0
        img = img.astype(np.uint8)
    else:
        # Đọc file ảnh thông thường (PNG/JPG)
        file_bytes = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Cân bằng mức xám CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img)
    
    return img, img_enhanced

def mock_ai_inference(img):
    """
    Hàm giả lập suy luận của DenseNet-121 & YOLOv5
    (Thay thế bằng hàm predict từ mô hình thật khi hoàn thiện model)
    """
    time.sleep(1.2) # Giả lập thời gian chạy AI (~1.2s)
    
    # Kết quả xác suất giả lập 15 bệnh lý
    results = {
        "Viêm phổi (Pneumonia)": 0.88,
        "Tràn dịch màng phổi": 0.65,
        "Lao phổi (TB)": 0.12,
        "Tim to (Cardiomegaly)": 0.05,
        "Tràn khí màng phổi": 0.02,
        "Xẹp phổi (Atelectasis)": 0.15,
        "Khối u / Nốt mờ": 0.08,
        "Phù phổi cấp": 0.01,
        "COPD / Tăng khí": 0.03,
        "Tổn thương hang": 0.04,
        "ARDS": 0.01,
        "Gãy xương sườn": 0.00,
        "Tràn khí MP áp lực": 0.00,
        "Thâm nhiễm": 0.22,
        "Xơ hóa phổi": 0.10
    }
    
    # Giả lập Bounding Box của YOLOv5 cho vùng Viêm phổi
    img_bbox = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    # Vẽ hộp giới hạn (Bounding Box) màu đỏ
    cv2.rectangle(img_bbox, (int(w*0.55), int(h*0.4)), (int(w*0.85), int(h*0.75)), (0, 0, 255), 3)
    cv2.putText(img_bbox, "Pneumonia 88%", (int(w*0.55), int(h*0.4)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Giả lập Grad-CAM Heatmap
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(img, (51, 51), 0), cv2.COLORMAP_JET)
    grad_cam = cv2.addWeighted(img_bbox, 0.7, heatmap, 0.3, 0)
    
    return results, img_bbox, grad_cam

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# ==========================================

# Header chính
st.markdown("<div class='main-header'>🫁 HỆ THỐNG AI PHÂN TÍCH ẢNH X-QUANG NGỰC<br><span style='font-size:16px; font-weight:normal; color:#555;'>Tự Động Phát Hiện Bệnh Lý Hô Hấp - Bệnh Viện Việt – Hàn Đà Nẵng</span></div>", unsafe_allow_html=True)

# Thanh Sidebar trái: Thông tin & Upload
with st.sidebar:
    st.header("📋 Tùy Chọn Đầu Vào")
    uploaded_file = st.file_uploader(
        "Tải ảnh X-quang (DICOM, PNG, JPG):", 
        type=["dcm", "png", "jpg", "jpeg"]
    )
    
    st.markdown("---")
    st.subheader("⚙️ Cấu Hình Mô Hình")
    threshold = st.slider("Ngưỡng cảnh báo bệnh (Threshold):", 0.0, 1.0, 0.5, 0.05)
    selected_model = st.selectbox("Mô hình phân loại:", ["DenseNet-121 (Khuyên dùng)", "ResNet-50", "EfficientNet-B4"])
    
    st.markdown("---")
    st.info("💡 **Ghi chú:** Hệ thống hỗ trợ xử lý file DICOM 12-bit từ PACS và tự động chuẩn hóa Cửa sổ phổi (Lung Window).")

# Luồng hiển thị chính
if uploaded_file is not None:
    # 1. Đọc và Tiền xử lý ảnh
    with st.spinner("Đang đọc và tiền xử lý ảnh (Lung Windowing & CLAHE)..."):
        img_raw, img_enhanced = process_dicom_or_image(uploaded_file)
    
    # Chia 2 cột hiển thị ảnh
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>📷 Ảnh X-Quang Gốc / Tiền Xử Lý</div>", unsafe_allow_html=True)
        st.image(img_enhanced, caption="Ảnh sau khi nâng cao độ tương phản (CLAHE)", use_container_width=True)
        
    # Nút bấm Phân tích AI
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH AI", type="primary", use_container_width=True):
        with st.spinner("AI đang phân tích các vi cấu trúc nhu mô phổi..."):
            results, img_bbox, grad_cam = mock_ai_inference(img_enhanced)
        
        with col2:
            st.markdown("<div class='sub-header'>🎯 Định Vị AI (YOLOv5 & Grad-CAM)</div>", unsafe_allow_html=True)
            view_option = st.radio("Chế độ hiển thị:", ["Bounding Box (YOLOv5)", "Bản đồ nhiệt Heatmap (Grad-CAM)"], horizontal=True)
            
            if view_option == "Bounding Box (YOLOv5)":
                st.image(img_bbox, caption="Vùng tổn thương được khoanh vùng tự động", use_container_width=True)
            else:
                st.image(grad_cam, caption="Grad-CAM thể hiện vùng tập trung của AI", use_container_width=True)
        
        st.markdown("---")
        
        # 2. Hiển thị Kết quả Phân loại Bệnh lý
        st.markdown("<div class='sub-header'>📊 BÁO CÁO KẾT QUẢ CHẨN ĐOÁN KỸ THUẬT Y SINH</div>", unsafe_allow_html=True)
        
        # Lọc danh sách bệnh vượt ngưỡng
        detected_diseases = {k: v for k, v in results.items() if v >= threshold}
        
        if detected_diseases:
            st.error(f"🚨 **CẢNH BÁO:** Phát hiện {len(detected_diseases)} dấu hiệu bệnh lý nghi vấn vượt ngưỡng {int(threshold*100)}%:")
            
            # Đưa vào Bảng dữ liệu Pandas
            df = pd.DataFrame(list(results.items()), columns=["Dấu hiệu Bệnh lý", "Xác suất (Probability)"])
            df["Xác suất (%)"] = (df["Xác suất (Probability)"] * 100).round(1)
            df["Trạng thái"] = df["Xác suất (Probability)"].apply(lambda x: "⚠️ Nghi vấn" if x >= threshold else "✅ Bình thường")
            
            # Sắp xếp xác suất giảm dần
            df = df.sort_values(by="Xác suất (Probability)", ascending=False).reset_index(drop=True)
            
            # Hiển thị bảng kết quả
            st.dataframe(
                df[["Dấu hiệu Bệnh lý", "Xác suất (%)", "Trạng thái"]], 
                use_container_width=True,
                height=300
            )
        else:
            st.success("✅ **KẾT LUẬN:** Không phát hiện dấu hiệu bất thường vượt ngưỡng cảnh báo.")
            
        # 3. Phản hồi của Bác sĩ (Active Learning)
        st.markdown("---")
        st.subheader("💬 Tương Tác Lâm Sàng & Phản Hồi (Active Learning)")
        feedback_col1, feedback_col2 = st.columns([3, 1])
        
        with feedback_col1:
            doctor_note = st.text_input("Ghi chú / Chẩn đoán đính chính của Bác sĩ:")
        with feedback_col2:
            st.write("") # Căn dòng
            st.write("")
            if st.button("💾 Xác nhận & Lưu Dữ liệu"):
                st.toast("Đã lưu phản hồi vào Candidate Pool phục vụ Huấn luyện lại (Retraining)!", icon="✅")

else:
    # Màn hình chờ khi chưa tải ảnh
    st.info("👈 Vui lòng chọn hoặc kéo thả file ảnh X-quang (.dcm, .png, .jpg) ở thanh bên trái để bắt đầu phân tích.")