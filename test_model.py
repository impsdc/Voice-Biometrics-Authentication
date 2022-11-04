from main_functions import *
import pyaudio
import wave
import cv2
import os
import pickle
import time
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output

# revognize current test.wav 

modelpath = "./gmm_models/"
FILENAME = "./test.wav"
gmm_files = [os.path.join(modelpath,fname) for fname in 
            os.listdir(modelpath) if fname.endswith('.gmm')]

models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]

speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
            in gmm_files]

if len(models) == 0:
    print("No Users in the database!!")


#read test file
sr,audio = read(FILENAME)

# extract mfcc features
vector = extract_features(audio,sr)
log_likelihood = np.zeros(len(models))

#checking with each model one by one
for i in range(len(models)):
    gmm = models[i]        
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()

print(speakers)
print(log_likelihood)
pred = np.argmax(log_likelihood)
identity = speakers[pred]
    

print( "Recognized as - ", identity)
