import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO models
paddy_model = YOLO("best1.pt")
chili_model = YOLO("best2.pt")
onion_model = YOLO("best3.pt")

# Streamlit page configuration
st.set_page_config(
    page_title="Tranquil Trails - Deteksi Objek",
    page_icon="🌾",
    layout="wide"
)

# Display logo on the main page
st.image("plants.png", width=200)

# Custom CSS for green leaf background
st.markdown(
    """
    <style>
        body {
            background-color: #228B22; /* Hijau daun */
            color: white; /* Teks menjadi putih */
        }
        [data-testid="stSidebar"] {
            background-color: #2E8B57; /* Hijau yang lebih gelap untuk sidebar */
        }
        .css-1v3fvcr {
            border: 2px dashed white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.image("plants.png", use_column_width=True)  # Logo in sidebar
st.sidebar.title("Tranquil Trails")
st.sidebar.header("Pilih Fitur")
feature_choice = st.sidebar.radio(
    "Pilih fitur yang ingin digunakan:",
    ("Deteksi Tanaman", "Chatbot")
)

if feature_choice == "Deteksi Tanaman":
    # Pilihan model deteksi
    st.sidebar.header("Pilih Model")
    model_choice = st.sidebar.radio(
        "Pilih jenis tanaman yang ingin dideteksi:",
        ("Padi", "Cabai", "Bawang")
    )

    # Map pilihan ke model
    if model_choice == "Padi":
        selected_model = paddy_model
    elif model_choice == "Cabai":
        selected_model = chili_model
    else:
        selected_model = onion_model

    # Main content layout
    st.title("🌾 Deteksi Objek untuk Tanaman")
    st.markdown(
        """
        <p style="font-size: 16px; text-align: center;">
            Pilih model deteksi di menu sebelah kiri, unggah gambar, dan lihat hasil deteksi.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Check if a model is selected
    st.markdown(f"### 🌱 Model Terpilih: **{model_choice}**")

    # Upload and process image
    uploaded_file = st.file_uploader(
        f"Unggah gambar {model_choice} untuk memulai deteksi:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display uploaded image
        st.image(img, channels="BGR", caption="📷 Gambar yang diunggah", use_column_width=True)

        # Run selected YOLO model on the image
        st.markdown("### 🔄 Memproses gambar dengan model yang dipilih...")
        results = selected_model(img)

        # Display results
        st.markdown("### ✅ Hasil Deteksi")
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="🔍 Deteksi Objek", use_column_width=True)
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #228B22; padding: 20px; text-align: center; border-radius: 10px;">
                <h3 style="color: #228B22;">Menunggu gambar diunggah</h3>
                <p style="color: #555;">Pilih dan unggah gambar untuk mendeteksi objek</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif feature_choice == "Chatbot":
    # Chatbot Interface
    st.title("🤖 Chatbot")
    st.markdown("Ajukan pertanyaan terkait pertanian atau penggunaan aplikasi ini.")

    # Input teks dari pengguna
    user_input = st.text_input("Ketik pesan Anda:", "")

    if user_input:
        # Logika respons chatbot sederhana (dapat diganti dengan model chatbot atau API lain)
        if "halo" in user_input.lower():
            response = "Halo! Bagaimana saya dapat membantu Anda hari ini?"
        elif "tanaman" in user_input.lower():
            response = "Saya dapat membantu mendeteksi padi, cabai, dan bawang. Apa yang ingin Anda ketahui lebih lanjut?"
        else:
            response = "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih detail?"

        # Tampilkan respons chatbot
        st.markdown(f"**Chatbot:** {response}")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #777;">Dibuat dengan ❤️ untuk mendukung petani Indonesia</p>
    """,
    unsafe_allow_html=True
)
