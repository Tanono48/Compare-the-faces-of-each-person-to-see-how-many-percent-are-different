import cv2
import streamlit as st

st.header('Compare the faces of each person to see how many percent are different')
st.markdown('--------------------------------')

image1_uploader = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
image2_uploader = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

percent_difference = None

if image1_uploader is not None and image2_uploader is not None:
    image1 = cv2.imread(str(image1_uploader))
    image2 = cv2.imread(str(image2_uploader))

    if image1 is not None and image2 is not None:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)

        percent_difference = 100 - (result[0][0] * 100)

if st.button('Run!!'):
    if image1_uploader is None or image2_uploader is None:
        st.write("Please upload both images.")
    else:
        st.write("The percentage difference between the two images is: {}".format(percent_difference))
        st.image(image1, caption='Image 1', use_column_width=True)
        st.image(image2, caption='Image 2', use_column_width=True)
        