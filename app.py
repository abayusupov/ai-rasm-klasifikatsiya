import streamlit as st
from fastai.vision.all import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Qurol va telefonni ajratuvchi model")

file = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])

model = load_learner('qurolmi_telefon_model.pkl')

if file:
    st.image(file)
    img = PILImage.create(file)

    pred, pr, prob = model.predict(img)
    if pred=='Handgun':
        st.success('Bu qurol')
    elif pred=='Mobile phone':
        st.success('Bu telefon')
    else:
        st.error('Bilmadim nimaligini :(')
    st.info(f' Ehtimollik: {prob[pr]*100:.1f}%')