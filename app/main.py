import streamlit as st #import libraries
import joblib
import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.preprocessing_training import clean_data
import plotly.graph_objects as go

def get_scaled_values(input_dict):

    data=clean_data()# fetch prepared data

    X = data.drop(['diagnosis'],axis=1)

    scaled_dict={}

    #sets values between 0 and 1
    for key, value in input_dict.items():
        max_val= X[key].max()
        min_val=X[key].min()
        scaled_value=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value

    return scaled_dict

def add_predictions(input_data):
    #load models
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    #change data to 1d array and apply scaler transformation
    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0]==0:
        st.write('Benign')
    else:
        st.write('Malignant')
    
    #print prediction probablities
    st.write("Probability of being benign: ", (round(model.predict_proba(input_array_scaled)[0][0],3)*100))
    st.write("Probability of being malignant: ",(round(model.predict_proba(input_array_scaled)[0][1],3)*100))

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")





def add_sidebar():
    #create sidebar header
    st.sidebar.header("Cell Nuclei Measurements")
    data = clean_data()# fetch prepared data

    #visual labels that correlate to each column in data
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict={}

    #loops through labels and assigns sliders with respective min and max values for each label
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()), # gets highest values from each column
            value=float(data[key].mean())
        )

    return input_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    #fetch data and assign categories to chart
    categories = ['Radius','Texture','Perimeter','Area',
                  'Smoothness','Compactness','Concavity',
                  'Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    #create multiple trace radar chart visualizing mean value, standard error and worst value
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    #update layout to show charts
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state='expanded'
    )

    input_data=add_sidebar()


    with st.container():
        st.title("Breast Cancer Predictor ")
        st.write("This app predicts whether a breast mass is benign or malignant based on the measurements it receives using a machine learning model. You can also update the measurements by hand using the sliders in the sidebar.")

    #takes a list of columns and a ratio of size between each column
    col1, col2=st.columns([4,1])

    with col1:# plot radar chart in first column and show predictions in second column
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)



if __name__=="__main__":
    main()
