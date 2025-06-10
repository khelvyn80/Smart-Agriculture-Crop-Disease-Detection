# train_hf_model.py

import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import (
	AutoImageProcessor,          # handles resizing & normalization
	TFAutoModelForImageClassification,
)
from tensorflow.keras.optimizers import Adam
# Hi

# ————————————————
# 0. Configurable paths and hyperparameters
DATA_DIR       = "ccmt_data"           # folder that contains `train/` and `val/`
MODEL_NAME     = "google/efficientnet-b0"   # Hugging Face checkpoint; can switch to a ResNet variant, e.g. "microsoft/resnet-50"
TF_SAVE_DIR    = "models/crop_disease_model_hf"
BATCH_SIZE     = 16
IMG_SIZE       = 224
NUM_EPOCHS     = 5
LEARNING_RATE  = 3e-5

# 1. Load your local image-folder dataset as an HF DatasetDict
#    The `imagefolder` loader expects subfolders named by class under train/ and val/
dataset = load_dataset(
	"imagefolder",
	data_dir=DATA_DIR,
	drop_labels=False  # keeps the “label” field as an int mapped from folder name
)
# dataset is a DatasetDict with splits: “train” and “val”

# 2. Get the list of class names in the same order (folder‐name sorted)
#    The dataset.features["label"].names list is sorted alphabetically by default.
class_names = dataset["train"].features["label"].names
print("Detected classes:", class_names)
# Should output something like:
# ["Cashew_Anthracnose", "Cashew_Gummosis", ..., "Tomato_Verticillium_Wilt"]

num_labels = len(class_names)  # should be 22

# 3. Load the HF Image Processor (resizes + normalizes to model’s expected range)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# 4. Preprocessing function that HF Trainer / TF .map() can use
def preprocess_train(examples):
	"""
	- `examples["image"]` is a PIL Image (loaded by `imagefolder`).
	- We run it through the processor: resizes to 224×224, normalizes RGB to [0,1].
	- Return pixel values plus “label”.
	"""
	images = [img.convert("RGB") for img in examples["image"]]
	#  Hugging Face “feature_extractor” expects PIL images in a list, and returns a dict:
	encodings = processor(images=images, return_tensors="np")
	# `encodings["pixel_values"]` shape: (batch_size, 3, IMG_SIZE, IMG_SIZE)
	# We need to transpose to (batch_size, IMG_SIZE, IMG_SIZE, 3) for TF:
	pixel_values = np.transpose(encodings["pixel_values"], (0, 2, 3, 1))
	return {"pixel_values": pixel_values, "label": examples["label"]}


# 5. Map preprocessing to train & val splits
#    We batch‐map them so that `pixel_values` will be NumPy arrays ready for TF.
train_ds = dataset["train"].with_transform(preprocess_train)
val_ds   = dataset["val"].with_transform(preprocess_train)

# 6. Convert HF Dataset to tf.data.Dataset for faster training
def convert_to_tf_dataset(hf_dataset, shuffle: bool):
	"""
	Takes an HF dataset with columns ["pixel_values", "label"] and converts to tf.data.
	"""
	# The HF dataset yields dictionaries: {"pixel_values": np.array(…, IMG_SIZE, IMG_SIZE, 3), "label": int}
	# We can map each example to a tuple (pixels, label).
	def gen():
		for example in hf_dataset:
			yield example["pixel_values"], example["label"]

	# Define output signature for tf.data.Dataset
	output_signature = (
		tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
		tf.TensorSpec(shape=(), dtype=tf.int64),
	)

	tf_dataset = tf.data.Dataset.from_generator(
		gen,
		output_signature=output_signature
	)
	if shuffle:
		tf_dataset = tf_dataset.shuffle(1000)
	tf_dataset = tf_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
	return tf_dataset

tf_train_ds = convert_to_tf_dataset(train_ds, shuffle=True)
tf_val_ds   = convert_to_tf_dataset(val_ds, shuffle=False)

# 7. Instantiate the TF version of EfficientNetB0 (or ResNet) with a new head for 22 classes
model = TFAutoModelForImageClassification.from_pretrained(
	MODEL_NAME,
	num_labels=num_labels,
	id2label={i: label for i, label in enumerate(class_names)},
	label2id={label: i for i, label in enumerate(class_names)},
)

# 8. Compile with optimizer, loss, and metrics
optimizer = Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 9. Train (fine‐tune)
history = model.fit(
	tf_train_ds,
	validation_data=tf_val_ds,
	epochs=NUM_EPOCHS,
)

# 10. Save the fine-tuned model in TensorFlow SavedModel format
#     so that it can be loaded by `tf.keras.models.load_model(...)` in your Streamlit app.
os.makedirs(TF_SAVE_DIR, exist_ok=True)
model.save_pretrained(TF_SAVE_DIR)  # this saves both config + weights
print(f"Model saved to {TF_SAVE_DIR}")
