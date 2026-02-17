# 🌸 Iris Flower Species Classifier

A web application that uses Deep Learning to classify Iris flowers into three species based on their sepal and petal measurements.

## 📋 Project Overview

This project implements a Deep Neural Network using TensorFlow/Keras to classify Iris flowers into:
- **Iris Setosa** 🌼
- **Iris Versicolor** 🌺
- **Iris Virginica** 🌸

The model takes four measurements as input:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the project directory:
```bash
cd DL_1
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Training the Model

Before running the web app, you need to train and save the model:

1. Open `project.ipynb` in Jupyter Notebook or VS Code
2. Run all cells in the notebook
3. Make sure to run the last cell that saves the model files:
   - `iris_model.h5` (trained neural network)
   - `scaler.pkl` (data scaler)
   - `label_encoder.pkl` (label encoder)

### Running the Application

Once the model is trained and saved, run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 🎯 How to Use

1. **Adjust the sliders** to input the measurements of your iris flower:
   - Sepal Length (4.0 - 8.0 cm)
   - Sepal Width (2.0 - 4.5 cm)
   - Petal Length (1.0 - 7.0 cm)
   - Petal Width (0.1 - 2.5 cm)

2. **Click the "Predict Species" button** to get the prediction

3. **View the results**:
   - Predicted species name
   - Confidence percentage
   - Probability distribution across all three species
   - Interactive bar chart showing probabilities

## 🧠 Model Architecture

The Deep Neural Network consists of:
- **Input Layer**: 4 features
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 8 neurons with ReLU activation
- **Output Layer**: 3 neurons with Softmax activation

**Training Configuration**:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 100
- Batch Size: 8

## 📊 Dataset

The model is trained on the famous [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) which contains 150 samples of iris flowers with equal distribution across three species.

## 🛠️ Technologies Used

- **Python**: Programming language
- **TensorFlow/Keras**: Deep Learning framework
- **Streamlit**: Web application framework
- **Scikit-learn**: Data preprocessing and model evaluation
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization

## 📁 Project Structure

```
DL_1/
├── Iris.csv              # Dataset
├── project.ipynb         # Training notebook
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── iris_model.h5         # Trained neural network model
├── scaler.pkl            # Data scaler for normalization
└── label_encoder.pkl     # Species label encoder
```

## 🎨 Features

- ✨ Interactive and user-friendly interface
- 📊 Real-time predictions with confidence scores
- 📈 Visual probability distribution
- 📱 Responsive design
- 🎯 High accuracy predictions
- 💡 Informative sidebar with model details

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Diganta**

---

Made with ❤️ using Deep Learning and Streamlit
