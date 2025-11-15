import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from fruit_info import fruit_info

# --- Load mô hình ---
model = tf.keras.models.load_model("model/fruit100_mobilenetv2 (1).h5")
input_size = model.input_shape[1:3]  # ví dụ (224, 224)

# Đọc file cũ
with open("classname.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Sắp xếp alphabet
class_names.sort()

# Ghi lại file
with open("classname_sorted.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")


# Kiểm tra số class
if len(class_names) != model.output_shape[-1]:
    st.warning("⚠️ Số class trong file classname.txt không khớp với số output của model!")
    
# --- Thông tin demo cho vài loại quả ---


# --- Streamlit giao diện ---
st.title(" What is this Fruit??!! 🍍🍌🍓")
st.write("Please upload pic of fruit to know its name and some information about it.")

def prepare_image(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload ảnh
uploaded_file = st.file_uploader("Upload pic of fruit", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded", use_container_width=True)
    
    # Chuẩn bị ảnh
    img_array = prepare_image(uploaded_file, input_size)
    
    # Dự đoán
    preds = model.predict(img_array)[0]
    
    # Lấy top 5 dự đoán
    top_indices = preds.argsort()[-5:][::-1]

    # Dự đoán chính
    pred_class = class_names[top_indices[0]]
    st.subheader(f" This fruit name is: **{pred_class.upper()}**")

    # Lấy thông tin từ dictionary
    info = fruit_info.get(pred_class.lower(), "Information not available.")

    # Thay dấu chấm bằng xuống dòng
    info_nice = info.replace(". ", ".\n")

    # Hiển thị
    st.text(info_nice)
