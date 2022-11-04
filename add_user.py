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
    
    name = input("Enter Name:")
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    
    source = "./voice_database/" + name
    
   
    os.mkdir(source)

    for i in range(5):
        audio = pyaudio.PyAudio()

        if i == 0:
            j = 3
            while j>=0:
                time.sleep(1.0)
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Speak your name in {} seconds".format(j))
                j-=1

        elif i ==1:
            time.sleep(2.0)
            print("Speak your name one more time")
            time.sleep(0.8)
        
        else:
            time.sleep(2.0)
            print("Speak your name one last time")
            time.sleep(0.8)

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

        print("recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # saving wav file of speaker
        waveFile = wave.open(source + '/' + str((i+1)) + '.wav', 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("Done")

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
            gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 5)
            gmm.fit(features)

            # saving the trained gaussian model
            pickle.dump(gmm, open(dest + name + '.gmm', 'wb'))
            print(name + ' added successfully') 
            
            features = np.asarray(())
            count = 0
        count = count + 1

if __name__ == '__main__':
    add_user()
