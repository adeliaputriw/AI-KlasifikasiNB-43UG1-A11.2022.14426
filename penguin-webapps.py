import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Web Apps Klasifikasi Penguin
Aplikasi berbasis Web untuk mengklasifikasi jenis penguin **Palmer Penguin**. \n
Palmer Penguin terdiri dari 3 jenis, yakni Adelie, Chinstrap, dan Gentoo.
Klasifikasi ini membutuhkan beberapa atribut, diantaranya  Island (pulau), Sex (jenis kelamin), Bill Length (panjang paruh), Bill Depth (kedalaman paruh), Flipper Length (panjang sirip), Body Mass (massa tubuh)
""")

st.header('Jenis Palmer Penguin')
st.write ("""
          Jenis Palmer Penguin ada 3, yakni
          Adelie, Chinstrap, dan Gentoo """)
img = Image.open('penguin.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.header('Paruh Penguin')
st.write("""
    Bill Length : Panjang paruh \n
    Bill Depth : Kedalaman paruh
""")

img2 = Image.open('culmen_depth.png')
img2 = img2.resize((700, 451))
st. image(img2, use_column_width=False)

## SIDEBAR BAGIAN KIRI UNTUK INPUTAN ##
st.sidebar.header('Parameter Inputan')

## UPLOAD FILE CSV UNTUK PARAMETER INPUTAN ##
upload_file = st.sidebar.file_uploader("Upload File CSV Anda", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male','female'))
        bill_length = st.sidebar.slider('Bill Length (mm)', 32.1,59.6,43.9)
        bill_depth = st.sidebar.slider('Bill Depth (mm)', 13.1,21.5,17.2)
        flipper_length = st.sidebar.slider('Flipper Length (mm)', 172.0,231.0,201.0)
        body_mass = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm' : bill_length,
                'bill_depth_mm' : bill_depth,
                'flipper_length_mm' : flipper_length,
                'body_mass_g' : body_mass,
                'sex' : sex}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

## MENGGABUNGKAN DATASET PENGUIN DENGAN INPUTAN ##    
penguin_raw = pd.read_csv('penguin.csv')
penguins = penguin_raw.drop(columns = ['species'])
df = pd.concat([inputan, penguins], axis=0)

## ENCODE UNTUK FITUR ORDINAL NUMBER (NUMERIK) ##
encode = ['island' , 'sex']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] ## AMBIL BARIS PERTAMA (input data user) ##

## MENAMPILKAN PARAMETER HASIL INPUTAN ##
st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Waiting for the CSV file to upload. Currently using input samples.')
    st.write(df)

## LOAD MODEL NBC ##
load_model = pickle.load(open('modelNBC_penguin.pkl', 'rb'))

## TERAPKAN NBC ##
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_penguin = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
label_penguin = np.array([0, 1, 2])
penguin_df = pd.DataFrame({'Kode Kelas': label_penguin, 'Jenis Penguin': jenis_penguin})
st.table(penguin_df)

## TAMPILAN PERHITUNGAN PROBABILITAS ##
st.subheader('Probabilitas Hasil Klasifikasi Penguin')
st.write(prediksi_proba)

## TAMPILAN HASIL KLASIFIKASI PENGUIN ##
st.subheader('Hasil Klasifikasi Penguin')
st.write(jenis_penguin[prediksi])