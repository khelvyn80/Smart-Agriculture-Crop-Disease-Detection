# app.py

import io
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import os

# ——————————————————————————————
# 1. Page Configuration
st.set_page_config(
	page_title="Crop Disease Detector",
	page_icon="🌾",
	layout="centered",
	initial_sidebar_state="expanded"
)

# ——————————————————————————————
# 2. Load the Model (at app startup)
MODEL_PATH = os.path.join("models", "crop_disease_model.h5")
@st.cache_resource(show_spinner=False)
def load_model(path):
	model = tf.keras.models.load_model(path)
	return model

model = load_model(MODEL_PATH)

# 22 class names in the same order you used in training
CLASS_NAMES = [
	"Cashew_Leaf_Spot",
	"Cashew_Anthracnose",
	"Cassava_Bacterial_Blight",
	"Cassava_Black_Zone",
	"Cassava_Green_Mite",
	"Cassava_Rust",
	"Maize_Gray_Leaf_Spot",
	"Maize_Northern_Leaf_Blight",
	"Maize_Leaf_Blight",
	"Maize_Maize_Lethal_Necrosis",
	"Tomato_Bacterial_spot",
	"Tomato_Black_Crinkle",
	"Tomato_Leaves_Flattening",
	# …and so on up to your 22 disease classes…
]

# ——————————————————————————————
# 3. Helper: Preprocess Uploaded Image
def preprocess_image(image_data):
	"""
	- image_data: raw bytes from uploaded file
	- Returns a numpy array shaped (1, 224, 224, 3), normalized.
	"""
	img = Image.open(io.BytesIO(image_data)).convert("RGB")
	img = img.resize((224, 224))
	arr = np.array(img) / 255.0  # normalize to [0,1]
	arr = np.expand_dims(arr, axis=0)
	return arr

# ——————————————————————————————
# 4. Main App UI
def main():
	st.title("🌾 Smart Agriculture: Crop Disease Detection")
	st.write(
		"""
		Upload an image of a leaf (cashew, cassava, maize, or tomato), 
		and the AI model will predict the disease class.
		"""
	)

	uploaded_file = st.file_uploader(
		label="Choose a leaf image (JPG/PNG)",
		type=["jpg", "jpeg", "png"]
	)

	if uploaded_file is not None:
		# Display the uploaded image
		image_bytes = uploaded_file.read()
		img = Image.open(io.BytesIO(image_bytes))
		st.image(img, caption="Uploaded Leaf", use_column_width=True)

		st.write("## Running prediction…")
		with st.spinner("Model is classifying..."):
			try:
				input_array = preprocess_image(image_bytes)
				preds = model.predict(input_array)[0]   # shape: (22,)
				idx = int(np.argmax(preds))
				confidence = float(preds[idx])
				predicted_class = CLASS_NAMES[idx]
			except Exception as e:
				st.error(f"Error during prediction: {e}")
				return

		# Display results
		st.success(f"**Disease:** {predicted_class}")
		st.info(f"**Confidence:** {confidence * 100:.1f}%")

		# (Optional) Add any advice mapping here
		advice_map = {
			"Cassava_Bacterial_Blight": "Apply copper-based bactericide early.",
			# … add advice for other classes …
		}
		advice = advice_map.get(predicted_class, "")
		if advice:
			st.write(f"**Advice:** {advice}")

	else:
		st.write("Please upload a leaf image to get started.")

if __name__ == "__main__":
	main()
