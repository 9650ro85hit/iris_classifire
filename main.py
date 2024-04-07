import streamlit as st
import numpy as np
import pandas as  pd
import joblib 
import sklearn

load_model = joblib.load("C:/Users/MYPC/streamlit_fldr/irismodel.pkl")
st.markdown("<h1 style='text-align: center; color: black;'>IrisExplorer</h1>", unsafe_allow_html=True)

# st.title("IrisExplorer")
data = []
col1, col2 = st.columns(2)
col4,col3 = st.columns(2)
with col1:
    SepalLengthCm = st.number_input("Enter SepalLength (Cm)", step=1)
    data.append(SepalLengthCm)
with col2:
    SepalWidthCm = st.number_input("Enter SepalWidth (Cm) ", step=1)
    data.append(SepalWidthCm)
with col3:
    PetalLengthCm = st.number_input("Enter PetalLength (Cm) ", step=1)
    data.append(PetalLengthCm)
with col4:
    PetalWidthCm = st.number_input("Enter PetalWidth (Cm) ", step=1)
    data.append(PetalWidthCm)


x_tst = np.array(data)
x_tst = x_tst.reshape(1,-1)
# print(x_tst.shape)

col = st.columns(1)
if st.button("Predict"):
    # print("the values are: ",SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    prd = load_model.predict(x_tst)
    imgName =""
    if prd ==0:
        imgName = "Setoasa"
    if prd == 1:
        imgName = "Versicolor"
    if prd == 2:
        imgName = "Verginica"
    st.markdown(f"<h3 style='text-align: center; color: blue;'>Predicted Image as {imgName}</h3>", unsafe_allow_html=True)

    # st.write("Predicted Image as : ",imgName)
    if prd == 0:
        st.image('C:/Users/MYPC/streamlit_fldr/Images/setosa.jpg', caption='Setosa image..')
        
    if prd == 1:
        st.image('C:/Users/MYPC/streamlit_fldr/Images/versicolor.jpg', caption='Versicolor image..')
        
    if prd == 2:
        st.image('C:/Users/MYPC/streamlit_fldr/Images/verginaica.jpg', caption='Verginica image..')
        
    

