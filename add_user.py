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
    
    new_name = input("Enter Name:")
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3

    VOICEPATH = "./voice_database/"
    
    source = VOICEPATH + new_name
    
   
    os.mkdir(source)

    for i in range(3):
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
    
    voice_dir = [ name for name in os.listdir(VOICEPATH) if os.path.isdir(os.path.join(VOICEPATH, name)) ]
    X = []
    Y = []
    for name in voice_dir:
        source = f"{VOICEPATH}{name}"
        for path in os.listdir(source):
                path = os.path.join(source, path)

                # reading audio files of speaker
                (sr, audio) = read(path)
                
                # extract 40 dimensional MFCC
                vector = extract_features(audio,sr)
                vector = vector.flatten()
                X.append(vector)
                Y.append(name)
    X = np.array(X, dtype=object)

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y_trans = le.transform(Y)
    clf = LogisticRegression(random_state=0).fit(X.tolist(), Y_trans)


    if os.path.isfile("gmm_models/voice_auth.gmm"): 
        os.remove("gmm_models/voice_auth.gmm")
    # saving model
    pickle.dump(clf, open('gmm_models/voice_auth.gmm', 'wb'))
    print(new_name + ' added successfully') 
    
    features = np.asarray(())
if __name__ == '__main__':
    add_user()
