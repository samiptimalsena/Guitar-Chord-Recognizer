import streamlit as st
import utils
import config
import model
import torch
import os

st.title("Guitar Chord Recognizer")

for i in range(4):
    st.write(" ")

st.image(config.TITLE_IMG)

for i in range(4):
    st.write(" ")

cols = st.beta_columns((1,1,1,2))

record_btn = cols[0].button("Record")
play_btn = cols[1].button("Play")
classify_btn = cols[2].button("Classify")
display_btn = cols[3].button("Display Melspectogram")

for i in range(2):
    st.write(" ")

slot = st.empty()

if record_btn:
    with st.spinner("Recording for 3 seconds"):
        utils.record()
    st.success("Recording Completed")

if play_btn:
    if os.path.exists(config.RECORDING_NPY):
        utils.play()
    else:
        slot.write("Please record sound first!!")

if classify_btn:
    if os.path.exists(config.RECORDING_NPY):
        model = model.load_model()
        mel_tensor = utils.create_tensor()
        pred = model(mel_tensor)
        pred_value = torch.argmax(pred)
        chord_name = config.CHORDS_MAPPING[pred_value.item()]
        
        slot.subheader(f"Chord : **{chord_name}**")
    else:
        slot.write("Please record sound first!!")  

if display_btn:
    if os.path.exists(config.RECORDING_NPY):
        utils.display_spectogram()
    else:
        slot.write("Please record sound first!!")  
