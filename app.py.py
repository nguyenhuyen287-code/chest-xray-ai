import streamlit as st
import cv2
import time
import pandas as pd
from fpdf import FPDF
import tempfile
import os
import hashlib
import numpy as np

# Hàm phụ trợ khử dấu tiếng Việt để tránh lỗi font khi xuất file PDF
def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOUUYaaaaeeeiioouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1: s += s0[s1.index(c)]
        else: s += c
    return s

# ==========================================
# 1. CẤU HÌNH TRANG WEB (CONFIG)
st.set_page_config(page_title="AI X-quang Phổi", layout="wide")
st.title("🫁 Hệ thống AI Hỗ trợ Phân tích X-quang Phổi")
st.markdown("Vui lòng tải ảnh X-quang của bệnh nhân lên để hệ thống xử lý.")

# Khôi phục hàm xử lý ảnh bị mất tên
def preprocess_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img)
    return img_enhanced

# Khôi phục nút tải ảnh giao diện
uploaded_file = st.file_uploader("Tải ảnh X-quang lên tại đây (định dạng: JPG, PNG, DCM)", type=["jpg", "png", "jpeg", "dcm", "dicom"])
def run_ai_inference(img):
    time.sleep(1.5) # Giả lập thời gian máy chủ AI xử lý
    
    # Đọc cấu trúc pixel của ảnh để tạo kết quả ĐỘNG (Ảnh khác nhau -> Kết quả khác nhau)
    img_hash = int(hashlib.md5(img.tobytes()).hexdigest(), 16)
    np.random.seed(img_hash % (2**32))
    
    disease_names = [
        "Viêm phổi (Pneumonia)", "Tràn dịch màng phổi", "Lao phổi (TB)", 
        "Tim to (Cardiomegaly)", "Tràn khí màng phổi", "Xẹp phổi (Atelectasis)", 
        "Khối u / Nốt mờ", "Phù phổi cấp", "COPD / Tăng khí", "Tổn thương hang", 
        "Thâm nhiễm mô kẽ", "Xơ hóa phổi"
    ]
    
    results = {}
    img_bbox = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape

    # Giả lập 40% ảnh đưa vào là Bình thường, 60% là Có bệnh
    is_healthy = np.random.rand() > 0.6 

    if is_healthy:
        # Nếu bình thường: Xác suất tất cả các bệnh đều thấp (< 40%)
        for name in disease_names:
            results[name] = np.random.uniform(0.01, 0.39)
        # Báo chữ xanh bình thường, KHÔNG vẽ Bounding Box
        cv2.putText(img_bbox, "NORMAL LUNG (No Findings)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Nếu có bệnh: Quét và đẩy xác suất 1 bệnh ngẫu nhiên lên cao (> 70%)
        for name in disease_names:
            results[name] = np.random.uniform(0.01, 0.45)
        main_disease = np.random.choice(disease_names)
        results[main_disease] = np.random.uniform(0.75, 0.98)
        
        # Giả lập YOLOv5 quét tìm tổn thương ở một vị trí ngẫu nhiên trong vùng phổi
        x1 = np.random.randint(int(w*0.1), int(w*0.5))
        y1 = np.random.randint(int(h*0.2), int(h*0.5))
        box_w = np.random.randint(int(w*0.15), int(w*0.35))
        box_h = np.random.randint(int(h*0.2), int(h*0.4))
        
        cv2.rectangle(img_bbox, (x1, y1), (x1+box_w, y1+box_h), (0, 0, 255), 3)
        cv2.putText(img_bbox, f"{main_disease} {results[main_disease]*100:.0f}%", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return results, img_bbox

# ==========================================
# 3. HÀM TẠO REPORT PDF (CHUẨN MẪU BỆNH VIỆT - HÀN)
# ... existing code ...
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Trang {self.page_no()} | AI Diagnostic System - KTYD Bách Khoa", 0, 0, 'C')

def generate_pdf_report(results, uploaded_filename, top_prediction, doctor_notes):
    pdf = MedicalPDFReport()
    pdf.add_page()
# ... existing code ...
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "NHẬN ĐỊNH LÂM SÀNG SƠ BỘ TỪ AI:", 0, 1, 'L')
    pdf.set_font("Arial", '', 10)
    
    if top_prediction[1] < 0.5:
        pdf.multi_cell(0, 5, "Mô hình AI nhận diện phổi bình thường, không có dấu hiệu bệnh lý nguy hiểm. Đề nghị Bác sĩ kiểm tra lại.")
    else:
        pdf.multi_cell(0, 5, f"AI nhận diện dấu hiệu cao nhất: {top_prediction[0]} (Độ tin cậy: {top_prediction[1]*100:.1f}%). Hệ thống đã kích hoạt YOLOv5 khoanh vùng tổn thương. Đề nghị Bác sĩ kiểm tra chi tiết hình ảnh.")
    pdf.ln(5)
    
    # --- THÊM PHẦN Ý KIẾN CỦA BÁC SĨ ---
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Ý KIẾN CHẨN ĐOÁN CỦA BÁC SĨ CHUYÊN KHOA:", 0, 1, 'L')
    pdf.set_font("Arial", 'I', 10)
    if doctor_notes.strip() == "":
        pdf.multi_cell(0, 5, "...................................................................................................................................................................................")
    else:
        # Khử dấu tiếng việt để FPDF cơ bản không bị lỗi Font
        clean_notes = remove_accents(doctor_notes)
        pdf.multi_cell(0, 5, clean_notes)
    pdf.ln(10)
    
    # Chữ ký
    pdf.set_font("Arial", 'B', 10)
    
if "results" in locals():
    table_data = []
    for disease, prob in results.items():
        status = "Bất thường" if prob > 0.5 else "Bình thường"
        table_data.append([disease, prob * 100, status])
        
    df = pd.DataFrame(table_data, columns=["Dấu hiệu Bệnh lý", "Xác suất (%)", "Trạng thái"])
    st.dataframe(df[["Dấu hiệu Bệnh lý", "Xác suất (%)", "Trạng thái"]], use_container_width=True, height=250)
        # --- KHU VỰC DÀNH CHO BÁC SĨ ---
    st.markdown("---")
    st.markdown("<div class='sub-header'>👨‍⚕️ TƯƠNG TÁC LÂM SÀNG (DÀNH CHO BÁC SĨ)</div>", unsafe_allow_html=True)
    doctor_notes = st.text_area("Nhập ý kiến chẩn đoán chuyên môn của Bác sĩ (Kết luận này sẽ được in trực tiếp vào Báo cáo PDF):", 
                                placeholder="Ví dụ: Bệnh nhân có tiền sử ho khan, hình ảnh X-quang cho thấy...")
    
    # --- TÍCH HỢP XUẤT REPORT PDF NGAY LẬP TỨC ---
    st.markdown("---")
    st.subheader("📥 Xuất Phiếu Kết Quả Lâm Sàng")
    top_disease = max(results.items(), key=lambda x: x[1])
            
    if st.button("Tạo File Báo Cáo PDF Chuẩn Bệnh Viện"):
            pdf_path = generate_pdf_report(results, uploaded_file.name, top_disease, doctor_notes)
            with open(pdf_path, "rb") as f:
                # Dòng dưới đây bắt buộc phải thụt lề (1 lần Tab) so với chữ 'with'
                st.download_button(
                    label="⬇️ Tải xuống Báo Cáo PDF",
                data=f,
                file_name="Bao_Cao_X_Quang_Phoi.pdf",
                mime="application/pdf"
            )
