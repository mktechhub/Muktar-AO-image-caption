import streamlit as st
import numpy as np
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import gdown
import os


# ---- Download models from Google Drive ----
def download_models():
    model_files = {
        "model (1).keras": "1Muy2szHvErl60Zi3gs0qdo6PISfK_nsq",
        "tokenizer (2).pkl": "1LoTcsh6A8CwW9N7DH1Br_Ib8NDChXZtA",
        "feature_extractor (1).keras": "1q5tDGe4Bu_lf7ynajjGeW_br3JbSwB1Q",
    }

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filename,
                quiet=False
            )


# ---- Generate Caption ----
def generate_and_display_caption(image_path, model_path, tokenizer_path,
                                  feature_extractor_path, max_length=42, img_size=224):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)


# ---- Streamlit App ----
def main():
    st.title("Afaan Oromo Image Captioner Model")
    st.write("Upload an image and generate a caption using the trained model.")

    # Download models on first run
    download_models()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path = "model (1).keras"
        tokenizer_path = "tokenizer (2).pkl"
        feature_extractor_path = "feature_extractor (1).keras"

        with st.spinner("Generating caption..."):
            generate_and_display_caption(
                "uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path
            )


if __name__ == "__main__":
    main()