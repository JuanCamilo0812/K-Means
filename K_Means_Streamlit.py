import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout='centered', page_title='Talento Tach ML', page_icon=("🐇"))

st.title('Clasificación de pinguinos')
steps=st.tabs(['Pinguinos','Arboles'])

with steps[0]:
    bill_length=st.number_input('Ingresa longitud del pico', min_value=0)
    bill_depth=st.number_input('Ingresa profundidad del pico', min_value=0)
    flipper_length=st.number_input('Ingresa la longitud de aleta', min_value=0)
    body_mass=st.number_input('Ingresa', min_value=0)
    island=st.selectbox('Escoja la isla',['Biscoe','Torgerson','Dream'])
    sex=st.selectbox('Escoja el sexo',['Femenino','Masculino'])