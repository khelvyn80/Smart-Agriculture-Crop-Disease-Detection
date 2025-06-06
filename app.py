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
# 2. Load the Keras-SavedModel (from our fine-tuning run)
#    Note: we point MODEL_PATH to the "keras_saved_model" folder inside the HuggingFace export.
MODEL_PATH = os.path.join("models", "crop_disease_model_hf", "keras_saved_model")

@st.cache_resource(show_spinner=False)
def load_model(path):
	"""
	This expects a TensorFlow SavedModel directory.
	After we fine-tuned and exported, we saved it with:
		model.tf_model.save(keras_dir, save_format="tf")
	So loading here gives us a tf.keras.Model we can call .predict() on.
	"""
	model = tf.keras.models.load_model(path)
	return model

model = load_model(MODEL_PATH)

# ——————————————————————————————
# 3. The 22 class names in EXACT alphabetical order (i.e. how `imagefolder` labeled them).
#    If you run `dataset["train"].features["label"].names`, you'll see this exact list.
CLASS_NAMES = [
	"Cashew_Anthracnose",
	"Cashew_Gummosis",
	"Cashew_Healthy",
	"Cashew_Leaf_Miner",
	"Cashew_Red_Rust",
	"Cassava_Bacterial_Blight",
	"Cassava_Brown_Spot",
	"Cassava_Green_Mite",
	"Cassava_Healthy",
	"Cassava_Mosaic",
	"Maize_Fall_Armyworm",
	"Maize_Grasshopper",
	"Maize_Healthy",
	"Maize_Leaf_Beetle",
	"Maize_Leaf_Blight",
	"Maize_Leaf_Spot",
	"Maize_Streak_Virus",
	"Tomato_Healthy",
	"Tomato_Leaf_Blight",
	"Tomato_Leaf_Curl",
	"Tomato_Septoria_Leaf_Spot",
	"Tomato_Verticillium_Wilt"
]

# ——————————————————————————————
# 4. Helper: Preprocess Uploaded Image (resize & normalize exactly as we did for training)
def preprocess_image(image_data):
	"""
	- image_data: raw bytes from uploaded file
	- Returns a numpy array shaped (1, IMG_SIZE, IMG_SIZE, 3), normalized to [0,1].
	"""
	IMG_SIZE = 224
	img = Image.open(io.BytesIO(image_data)).convert("RGB")
	img = img.resize((IMG_SIZE, IMG_SIZE))
	arr = np.array(img) / 255.0       # normalize to [0,1]
	arr = np.expand_dims(arr, axis=0) # shape: (1, 224, 224, 3)
	return arr

# ——————————————————————————————
# 5. Main App UI
def main():
	st.title("🌾 Smart Agriculture: Crop Disease Detection")
	st.write(
		"""
		Upload an image of a crop leaf (cashew, cassava, maize, or tomato), 
		and the AI model will predict which disease (or healthy) class it belongs to.
		"""
	)

	uploaded_file = st.file_uploader(
		label="Choose a leaf image (JPG/PNG)",
		type=["jpg", "jpeg", "png"]
	)

	if uploaded_file is not None:
		# 5a. Display the uploaded image
		image_bytes = uploaded_file.read()
		img = Image.open(io.BytesIO(image_bytes))
		st.image(img, caption="Uploaded Leaf", use_column_width=True)

		st.write("## Running prediction…")
		with st.spinner("Model is classifying..."):
			try:
				input_array = preprocess_image(image_bytes)       # (1, 224, 224, 3)
				logits = model.predict(input_array)              # shape: (1, 22)
				preds = tf.nn.softmax(logits, axis=-1).numpy()[0] # convert to probabilities
				idx = int(np.argmax(preds))
				confidence = float(preds[idx])
				predicted_class = CLASS_NAMES[idx]
			except Exception as e:
				st.error(f"Error during prediction: {e}")
				return

		# 5b. Display results
		st.success(f"**Disease:** {predicted_class}")
		st.info(f"**Confidence:** {confidence * 100:.1f}%")

		# 5c. (Optional) Advice lookup
		advice_map = {
			"Cashew_Anthracnose": "Prune infected branches and apply copper fungicide.",
			"Cashew_Gummosis": "Remove oozing parts and treat with protective fungicide.",
			"Cashew_Healthy": "No action needed; plant is healthy.",
			"Cashew_Leaf_Miner": "Use neem-based insecticide early in infestation.",
			"Cashew_Red_Rust": "Apply recommended rust fungicide; improve air circulation.",
			"Cassava_Bacterial_Blight": "Apply copper-based bactericide; destroy badly infected stems.",
			"Cassava_Brown_Spot": "Spray recommended fungicide every 14 days.",
			"Cassava_Green_Mite": "Introduce predatory mites and use neem oil.",
			"Cassava_Healthy": "No action needed; plant is healthy.",
			"Cassava_Mosaic": "Use mosaic-resistant varieties and control whitefly.",
			"Maize_Fall_Armyworm": "Use early-stage biological control (Bacillus thuringiensis).",
			"Maize_Grasshopper": "Apply insecticide recommended by extension officers.",
			"Maize_Healthy": "No action needed; plant is healthy.",
			"Maize_Leaf_Beetle": "Introduce natural predators and use neem-based sprays.",
			"Maize_Leaf_Blight": "Apply fungicide early and practice crop rotation.",
			"Maize_Leaf_Spot": "Remove infected leaves and apply copper fungicide.",
			"Maize_Streak_Virus": "Use resistant hybrids and control vector (leafhoppers).",
			"Tomato_Healthy": "No action needed; plant is healthy.",
			"Tomato_Leaf_Blight": "Apply strobilurin-based fungicide at first sign.",
			"Tomato_Leaf_Curl": "Use reflective mulch and control whiteflies.",
			"Tomato_Septoria_Leaf_Spot": "Remove lower leaves and apply chlorothalonil.",
			"Tomato_Verticillium_Wilt": "Use resistant varieties and rotate crops."
		}
		advice = advice_map.get(predicted_class, "")
		if advice:
			st.write(f"**Advice:** {advice}")

	else:
		st.write("Please upload a leaf image to see predictions.")

if __name__ == "__main__":
	main()
