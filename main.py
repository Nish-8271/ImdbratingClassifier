import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

### Mapping of word indexes back to words (for understanding )
word_index=imdb.get_word_index()

reverse_word_index={value:key for key,value in word_index.items()}

# Load the preTrained model with reLU activation function
from tensorflow.keras.layers import InputLayer

# Define a robust loading function that handles the 'batch_shape' issue across Keras versions
def safe_load_model(model_path):
    try:
        # Try standard loading first
        return load_model(model_path)
    except Exception as e:
        if "batch_shape" in str(e):
            # If batch_shape error occurs, provide a custom InputLayer that ignores it
            def fixed_input_layer(**kwargs):
                kwargs.pop('batch_shape', None)
                return InputLayer(**kwargs)
            return load_model(model_path, custom_objects={'InputLayer': fixed_input_layer})
        raise e

model = safe_load_model('imdb_review_rnn.h5')

# Step 2: Helper Functions
# Functions to decode reviews back to text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input 
def prepocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]  # +3 to account for reserved indices
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

### Prediction Function
def predict_sentiment(review):
    preprocessed_input=prepocess_text(review)

    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] < 0.5 else 'Negative'
    return sentiment,prediction[0][0]

## Strimlit app
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negative.")

#User input 
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=prepocess_text(user_input)
    ## Make Prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]<0.5 else 'Negative'

    # Display the result 
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write("Please enter a review.")