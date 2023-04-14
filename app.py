import streamlit as st
import numpy as np
import cv2
import av
import os
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer


@st.cache_resource
def load_model():
    model = keras.models.load_model('./waste_classifier_v2')
    return model


INDEX_TO_CLASS = {0: 'cardboard', 1: 'glass', 2: 'metal',
                  3: 'organic', 4: 'paper', 5: 'plastic', 6: 'trash'}

def callback_predict(frame):
    pic = frame.to_ndarray(format='rgb24')
    copy = cv2.resize(pic, (128, 128)) / 255

    pred = model.predict(copy.reshape(1, 128, 128, 3), verbose=0)
    probability = pred.max()
    index = np.argmax(pred)
    label = INDEX_TO_CLASS.get(index).capitalize()

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    pic = cv2.putText(pic, '%s: %.2f%%' % (label, probability * 100), org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(pic)

def random_images():
    url = './dataset-resized/'

    def random_choice(type):
            path = url + type
            file = np.random.choice(os.listdir(path))
            pic = cv2.imread(os.path.join(path, file))

            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            copy = cv2.resize(pic, (128, 128)) / 255

            pred = model.predict(copy.reshape(1, 128, 128, 3), verbose=0)
            probability = pred.max()
            index = np.argmax(pred)
            label = INDEX_TO_CLASS.get(index).capitalize()

            return pic, label, probability
    
    TYPES = list(INDEX_TO_CLASS.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        type1 = np.random.choice(TYPES)
        pic1, label1, probability1 = random_choice(type1)

        st.markdown(f'{label1}: {probability1 * 100 : .2f}%')

        pic1 = cv2.resize(pic1, (512, 512))
        st.image(pic1, caption=type1.capitalize())
    with col2:
        type2 = np.random.choice(TYPES)
        pic2, label2, probability2 = random_choice(type2)

        st.markdown(f'{label2}: {probability2 * 100 : .2f}%')

        pic2 = cv2.resize(pic2, (512, 512))
        st.image(pic2, caption=type2.capitalize())
    with col3:
        type3 = np.random.choice(TYPES)
        pic3, label3, probability3 = random_choice(type3)

        st.markdown(f'{label3}: {probability3 * 100 : .2f}%')

        pic3 = cv2.resize(pic3, (512, 512))
        st.image(pic3, caption=type3.capitalize())
    with col4:
        type4 = np.random.choice(TYPES)
        pic4, label4, probability4 = random_choice(type4)

        st.markdown(f'{label4}: {probability4 * 100 : .2f}%')
        
        pic4 = cv2.resize(pic4, (512, 512))
        st.image(pic4, caption=type4.capitalize())


model = load_model()

st.title('Waste Classification App')
st.markdown('Portable AI classifier can be integrated into small-size cameras with accuracy over 90% identifying waste images.')
st.markdown('---------------------------------------------------------------')


mode = st.sidebar.selectbox(
    'Mode', ['Random Images', 'Import Images', 'Camera'])
if mode == 'Random Images':
    if st.button('Generate'):
        random_images()
elif mode == 'Import Images':
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        pic = cv2.imdecode(file_bytes, 1)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        copy = cv2.resize(pic, (128, 128)) / 255
        pred = model.predict(copy.reshape(1, 128, 128, 3), verbose=0)

        pred = model.predict(copy.reshape(1, 128, 128, 3), verbose=0)
        probability = pred.max()
        index = np.argmax(pred)
        label = INDEX_TO_CLASS.get(index).capitalize()

        st.markdown(f'{label}: {probability * 100 : .2f}%')

        pic = cv2.resize(pic, (384, 384))
        st.image(pic, caption="Uploaded Image")
elif mode == 'Camera':
    webrtc_streamer(key="example", video_frame_callback=callback_predict)
    # access_camera()


st.markdown('---------------------------------------------------------------')
st.write('Created by Tri Cao Chanh 2023')
