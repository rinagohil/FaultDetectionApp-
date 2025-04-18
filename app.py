import streamlit as st
import numpy as np
import scipy.io
from scipy.fft import fft
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import gdown
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# ------------------ Database Setup ------------------
Base = declarative_base()
engine = create_engine('sqlite:///users.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)
session = Session()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    hashed_password = Column(String(255))

Base.metadata.create_all(engine)

def create_user(username, password):
    if session.query(User).filter(User.username == username).first():
        return False
    hashed_password = pwd_context.hash(password)
    user = User(username=username, hashed_password=hashed_password)
    session.add(user)
    session.commit()
    return True

def authenticate_user(username, password):
    user = session.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return False
    return True

# ------------------ Authentication UI ------------------
def auth_page():
    st.title("Fault Detection System - Authentication")
    
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    auth_mode = st.radio("Choose action:", 
                       ("Login", "Sign Up"),
                       index=0 if st.session_state.auth_mode == 'login' else 1)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
        if st.button("Sign Up"):
            if create_user(username, password):
                st.success("Account created successfully! Please login.")
                st.session_state.auth_mode = 'login'
            else:
                st.error("Username already exists")

# ------------------ Main Application ------------------
def main_app():
    st.title(f"Fault Detection using FFT and AlexNet - Welcome {st.session_state.username}")

    # Download model if not exists
    if not os.path.exists("model.weights.h5"):
        url = "https://drive.google.com/uc?id=1AbcDefGHIJKLmnopQRStuvWXyz"
        gdown.download(url, "model.weights.h5", quiet=False)

    # Load the model
    def load_model():
        with open("model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.weights.h5")
        return model

    # Extract FFT features from signal
    def extract_fft_features(signal, n_features=50):
        fft_values = np.abs(fft(signal))
        half_length = len(fft_values) // 2
        fft_features = fft_values[:half_length]
        if len(fft_features) < n_features:
            padded = np.zeros(n_features)
            padded[:len(fft_features)] = fft_features
            return padded
        else:
            return fft_features[:n_features]

    # Load signal from .mat file
    def load_signal_from_mat(file, variable_name=None):
        mat_data = scipy.io.loadmat(file)
        if variable_name:
            signal = mat_data[variable_name]
        else:
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            signal = mat_data[keys[0]]
        return signal.flatten()

    # Main application UI
    uploaded_file = st.file_uploader("Upload a .mat file", type=["mat"])

    if uploaded_file is not None:
        try:
            signal = load_signal_from_mat(uploaded_file)
            features = extract_fft_features(signal)
            features_reshaped = features.reshape(1, 5, 10, 1)

            model = load_model()
            prediction = model.predict(features_reshaped)
            pred_class = np.argmax(prediction)

            class_names = ["Healthy", "Faulty"]
            st.success(f"Prediction: **{class_names[pred_class]}**")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ------------------ Main Execution ------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    auth_page()