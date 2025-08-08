\# Breast Cancer Detection App



\## Overview



The Breast Cancer Detection App is an interactive web application designed to assist medical professionals in diagnosing breast cancer. This app utilizes machine learning to predict whether a cell cluster is benign or malignant based on a set of measurements.



This project is intended for use as practice \& an educational tool for biotechnology and the application of AI in healthcare.



\## Features



Streamlit-hosted UI with labeled sliders in the sidebar to customize input measurements

Real-time predictions with a probability score indicating the likelihood of the cell cluster being benign or malignant

Radar chart visualization to help medical professionals to better understand input data.



\## How It Works



The app uses a trained machine learning model to analyze metrics including but not limited to:



Radius

Texture

Smoothness

Compactness

Area



\## Tools \& Technologies



Streamlit

scikit-learn

pandas

numpy

joblib

plotly



\## Installation and Running



Clone repository



Install dependencies : pip install -r requirements.txt



Run the Streamlit app : streamlit run app/main.py



Input metrics using sliders



\## Next Possible Steps



Develop feature to quickly implement metrics

Improve visual features of application

