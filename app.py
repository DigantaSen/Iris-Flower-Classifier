import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Suppress absl warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.models import load_model

# Load the saved model, scaler, and encoder
@st.cache_resource
def load_models():
    model = load_model('iris_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, scaler, encoder

model, scaler, encoder = load_models()

# App title and description
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Species Classifier")
st.markdown("### Predict the species of Iris flower using Deep Learning")
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📏 Sepal Measurements")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    st.markdown("#### 🌺 Petal Measurements")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

st.markdown("---")

# Display input summary
st.markdown("### 📊 Your Input Summary")
input_df = pd.DataFrame({
    'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
})
st.table(input_df)

# Prediction button
if st.button("🔍 Predict Species", type="primary"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict(input_scaled, verbose=0)
    prediction_class = np.argmax(prediction_proba, axis=1)
    
    # Decode the prediction
    predicted_species = encoder.inverse_transform(prediction_class)[0]
    confidence = np.max(prediction_proba) * 100
    
    # Display results
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")
    
    # Create columns for result display
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Predicted Species", predicted_species.upper())
    
    with res_col2:
        st.metric("Confidence", f"{confidence:.2f}%")
    
    # Display probability for each class
    st.markdown("### 📈 Prediction Probabilities")
    species_names = encoder.classes_
    prob_df = pd.DataFrame({
        'Species': species_names,
        'Probability (%)': prediction_proba[0] * 100
    }).sort_values('Probability (%)', ascending=False)
    
    # Create bar chart
    st.bar_chart(prob_df.set_index('Species'))
    
    # Show detailed probabilities
    st.dataframe(prob_df, width='stretch')
    
    # Add flower emoji based on species
    species_emoji = {
        'Iris-setosa': '🌼',
        'Iris-versicolor': '🌺',
        'Iris-virginica': '🌸'
    }
    
    st.success(f"{species_emoji.get(predicted_species, '🌸')} The flower is predicted to be **{predicted_species}** with {confidence:.2f}% confidence!")

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    This app uses a Deep Learning Neural Network to classify Iris flowers into three species:
    - **Iris Setosa** 🌼
    - **Iris Versicolor** 🌺
    - **Iris Virginica** 🌸
    
    The model is trained on the famous Iris dataset and uses measurements of sepals and petals to make predictions.
    """)
    
    st.markdown("### 📚 Model Details")
    st.markdown("""
    - **Architecture**: Deep Neural Network
    - **Layers**: 3 Dense layers
    - **Activation**: ReLU, Softmax
    - **Optimizer**: Adam
    - **Loss**: Categorical Crossentropy
    """)
    
    st.markdown("### 🔗 Dataset")
    st.markdown("[Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)")
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Diganta")
