import streamlit as st
from PIL import Image
import cv2
import pywt
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl","rb"))

# ---- Title & Icon ----
st.set_page_config(page_title = "Classifier Page", page_icon = ":globe_with_meridians:",layout = "wide")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            img { height : 150px; width : 150px; border-radius : 50%;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            button[title="View fullscreen"] {visibility: hidden;}
            .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
            .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def load_image(image_file):
	img = Image.open(image_file).convert('RGB')
	return np.array(img)

def get_cropped_image_if_2_eyes(img):

    face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def get_feature(img):
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

    len_image_array = 32*32*3 + 32*32

    final = combined_img.reshape(1,len_image_array).astype(float)
    return final


st.text('')
st.markdown("<h1 style='text-align: center;'>Sports Person Classifier</h1>", unsafe_allow_html=True)
# st.title('Sports Person Classifier')
st.text('')
st.text('')

col1 ,col2, col3, col4, col5 = st.columns(5)
img_dict = {
    'Lionel Messi' : 
    'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Lionel_Messi_20180626.jpg/330px-Lionel_Messi_20180626.jpg',
    'Maria Sharapova' :
    'https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Maria_Sharapova_Australian_Open_Players%27_Party_2015.jpg/330px-Maria_Sharapova_Australian_Open_Players%27_Party_2015.jpg',
    'Roger Federer' : 
    'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Federer_WM16_%2837%29_%2828136155830%29.jpg/330px-Federer_WM16_%2837%29_%2828136155830%29.jpg',
    'Serena Williams' : 
    'https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Serena-Smiling-2020.png/375px-Serena-Smiling-2020.png',
    'Virat Kolhi' : 
    'https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Shri_Virat_Kohli_for_Cricket%2C_in_a_glittering_ceremony%2C_at_Rashtrapati_Bhavan%2C_in_New_Delhi_on_September_25%2C_2018_%28cropped%29.JPG/330px-Shri_Virat_Kohli_for_Cricket%2C_in_a_glittering_ceremony%2C_at_Rashtrapati_Bhavan%2C_in_New_Delhi_on_September_25%2C_2018_%28cropped%29.JPG'        
}

ans_value = { 0 : 'Lionel Messi', 1 : 'Maria Sharapova', 2 : 'Roger Federer', 3 : 'Serena Williams', 4 : 'Virat Kolhi'}

with col1:
    st.image(list(img_dict.values())[0])
    st.subheader(list(img_dict.keys())[0])

with col2:
    st.image(list(img_dict.values())[1])
    st.subheader(list(img_dict.keys())[1])

with col3:
    st.image(list(img_dict.values())[2])
    st.subheader(list(img_dict.keys())[2])

with col4:
    st.image(list(img_dict.values())[3])
    st.subheader(list(img_dict.keys())[3])

with col5:
    st.image(list(img_dict.values())[4])
    st.subheader(list(img_dict.keys())[4])

st.text('')
st.text('')
image = st.file_uploader('Upload the photo')

col1, col2 = st.columns(2)

with col1:
    if image:
        st.image(image)

with col2:
    if image:
        image_predict = load_image(image)
        roi_image = get_cropped_image_if_2_eyes(image_predict)
        if roi_image is None: 
            st.warning("Please Upload Another Photo") 
        else:
            feature = get_feature(roi_image)
            result = model.predict(feature)
            probability_list = np.around(model.predict_proba(feature)*100,2).tolist()[0]
            
            data = {'Players':list(ans_value.values()),
                    'Probability Value':[p_value for p_value in probability_list]}
            df = pd.DataFrame(data)
            st.table(df)   

            st.success("So, Predicted Person is " + ans_value[result[0]])      