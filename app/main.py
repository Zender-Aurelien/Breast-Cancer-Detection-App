import streamlit as st
import joblib
import pandas as pd

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state='expanded'
    )

    #st.write to create a <p> element
    #st.write("Hello World!")


    with st.container():
        st.title("Breast Cancer Predictor ")
        st.write("description placeholder")



if __name__=="__main__":
    main()