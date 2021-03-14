import os

TITLE_IMG = os.path.abspath('../utils/title_image.jpg')
RECORDING_NPY = os.path.abspath('../utils/recording.npy')
RECORDING_WAV = os.path.abspath('../utils/recording.wav')
MODEL_PATH = os.path.abspath('../model/model_20ep.pt')

CHORDS_MAPPING = {
    0 : 'c',
    1 : 'dm',
    2 : 'bm',
    3 : 'g',
    4 : 'em',
    5 : 'a',
    6 : 'am',
    7 : 'd',
    8 : 'f',
    9 : 'e'
}