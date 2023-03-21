import pandas as pd
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import json

from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie


st.set_page_config(page_title="DigitX",layout="wide")

model = load_model('model.h5')

result = None


#Animation files load funcion
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
zero = load_lottiefile("0.json")
one = load_lottiefile("1.json")
two = load_lottiefile("2.json")
three = load_lottiefile("3.json")
four = load_lottiefile("4.json")
five = load_lottiefile("5.json")
six = load_lottiefile("6.json")
seven = load_lottiefile("7.json")
eight = load_lottiefile("8.json")
nine = load_lottiefile("9.json")
empty = load_lottiefile('ai.json')



st.title("Hand Written digit recognizor")
st.subheader("Integrated with Animation's & Build with Streamlit")
st.write("AI Model Trained with Nueral Networks")


col1,col2,col3 = st.columns(3)
with col1:
    st.subheader("Draw a Number")
    canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color='000000',
    height=300,width=300,
    drawing_mode='freedraw',
    key='canvas',
 
    
    )
    
       




    with col2:
        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
            img_rescalling = cv2.resize(img,(300,300),interpolation=cv2.INTER_NEAREST)
            st.subheader("Input Image")
            st.image(img_rescalling)
            if st.button('predict'):
                test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                pred = model.predict(test_x.reshape(1,28,28))
                result = np.argmax(pred[0])
                st.header(result)

        with col3:
            if result==None:
                st.subheader("Ready to Predict")
                st_lottie(empty, speed=1, reverse=False, quality="high",loop=True, height=500,width=500)
            elif (result == 0):
                st.subheader("Predicted Number")
                st_lottie(zero, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 1):
                st.subheader("Predicted Number")
                st_lottie(one, speed=1, reverse=False, quality="high",loop=False, height=500,width=400)
            elif (result == 2):
                st.subheader("Predicted Number")
                st_lottie(two, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 3):
                st.subheader("Predicted Number")
                st_lottie(three, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 4):
                st.subheader("Predicted Number")
                st_lottie(four, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 5):
                st.subheader("Predicted Number")
                st_lottie(five, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 6):
                st.subheader("Predicted Number")
                st_lottie(six, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 7):
                st.subheader("Predicted Number")
                st_lottie(seven, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 8):
                st.subheader("Predicted Number")
                st_lottie(eight, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)
            elif (result == 9):
                st.subheader("Predicted Number")
                st_lottie(nine, speed=1, reverse=False, quality="high",loop=False, height=500,width=500)

            


        
            
               
