import pyaudio
import wave
import cv2
import os
import pickle
import time
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output

from main_functions import *

def add_user():
    
    name = "antoine"
    source = "./voice_database/" + name
    
    dest =  "./gmm_models/"
    
    count = 1

    for path in os.listdir(source):
        path = os.path.join(source, path)

        features = np.array([])
        
        # reading audio files of speaker
        (sr, audio) = read(path)
        
        # extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
            
        # when features of 3 files of speaker are concatenated, then do model training
        if count == 5:    
            gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 5)
            gmm.fit(features)

            # saving the trained gaussian model
            pickle.dump(gmm, open(dest + name + '.gmm', 'wb'))
            print(name + ' added successfully') 
            
            features = np.asarray(())
            count = 0
        count = count + 1

if __name__ == '__main__':
    add_user()
