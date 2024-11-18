import streamlit as st
import tensorflow as tf
import re

# Load the trained model
model = tf.keras.models.load_model("models/toxic_tweet_model.h5")

# Preprocessing function
def clean_text(text):
    return re.sub(r"http\S+|@\w+|#\w+|[^\w\s]", "", text).lower().strip()

# Define the Streamlit app
def main():
    st.title("Toxic Tweet Classifier")
    st.write("A simple app to classify whether a tweet is toxic or non-toxic.")

    # Text input
    user_input = st.text_area("Enter a tweet below:", placeholder="Type your tweet here...")
    
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a valid tweet!")
        else:
            # Preprocess the input
            processed_text = clean_text(user_input)

            # Vectorize the text
            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=20000, output_sequence_length=10
            )
            vectorizer.adapt([processed_text])  # Adapt on the input text
            input_sequence = vectorizer([processed_text])

            # Predict using the model
            prediction = model.predict(input_sequence)
            label = "Toxic" if prediction > 0.5 else "Non-toxic"

            # Display the result
            st.write(f"### Prediction: **{label}**")
            st.write(f"**Confidence Score:** {prediction[0][0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()
