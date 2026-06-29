import streamlit as st
import tensorflow as tf
import pickle

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

# Load model
model = tf.keras.models.load_model("spam_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("📧 Spam Email Detection")
st.write("Enter an email/message to check whether it is Spam or Ham.")

message = st.text_area("Enter your message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        seq = tokenizer.texts_to_sequences([message])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

        prediction = model.predict(padded)

        if prediction[0][0] > 0.5:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam (Ham)")