# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# --- Page Configuration ---
st.set_page_config(
    page_title="Devanagari Character Recognizer",
    page_icon="✍️",
    layout="wide"
)

# --- Load the Pre-trained Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained CNN model."""
    try:
        model = tf.keras.models.load_model('devanagari_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 'devanagari_model.h5' file is in the same directory and you have run the training script first.")
        return None

model = load_model()

# --- Devanagari Character Mapping ---
# CORRECTED: This list now accurately matches the 46 classes sorted alphabetically 
# by the training script, resolving the IndexError.
CHAR_MAP = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'adna', 'ba', 'bha', 'cha', 'chha', 'da', 'dha', 'ga', 'gha',
    'gya', 'ha', 'ja', 'jha', 'ka', 'kha', 'kna', 'ksha', 'la',
    'ma', 'na', 'pa', 'pha', 'ra', 'sa', 'sha', 'shat', 'ta',
    'tha', 'tra', 'waw', 'yaw', 'yna'
]


# --- UI Elements ---
st.title("✍️ Devanagari Handwritten Character Recognition")
st.markdown("""
Draw a single Devanagari character or digit on the canvas below and click 'Predict' to see the model's prediction.
This application uses a **Convolutional Neural Network (CNN)**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if st.button('Predict Character'):
    if model is not None and canvas_result.image_data is not None:
        # --- Preprocess the drawn image ---
        img = canvas_result.image_data.astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img_resized = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA)
        img_reshaped = img_resized.reshape(1, 32, 32, 1)
        # Normalization is now part of the model, so we don't do it here.

        # --- Make Prediction ---
        prediction = model.predict(img_reshaped)
        predicted_index = np.argmax(prediction)
        
        # This line should now work without error
        predicted_char = CHAR_MAP[predicted_index]
        confidence = np.max(prediction) * 100

        with col2:
            st.subheader("Prediction Result")
            st.metric(label="Predicted Character", value=f"{predicted_char}")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
            st.write("---")
            st.write("Top 5 Predictions:")
            top_indices = np.argsort(prediction[0])[-5:][::-1]
            prob_df = pd.DataFrame({
                'Character': [CHAR_MAP[i] for i in top_indices],
                'Probability': [f"{prediction[0][i]*100:.2f}%" for i in top_indices]
            })
            st.dataframe(prob_df)

    else:
        st.warning("Please draw a character on the canvas first.")

st.info("""
**How to Run This App:**
1.  **Download & Unzip:** Download the dataset from Kaggle and unzip it. Place the `DevanagariHandwrittenCharacterDataset` folder next to your Python files.
2.  **Install Libraries:** `pip install streamlit pandas numpy tensorflow scikit-learn opencv-python streamlit-drawable-canvas`
3.  **Train Model:** Run `python train_model.py` (the new, faster version). This will create the `devanagari_model.h5` file.
4.  **Run App:** Once training is complete, run `streamlit run app.py`.
""")
