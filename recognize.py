import pyaudio
import wave
import cv2
import os
import pickle
import time
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output

from main_functions import *

def recognize():
    # Voice Authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "./test.wav"
    MODEL = "gmm_models/voice_auth.gmm"
    VOICEPATH = "./voice_database/"
    VOICENAMES = [ name for name in os.listdir(VOICEPATH) if os.path.isdir(os.path.join(VOICEPATH, name)) ]

    
    try:
        while True:
            audio = pyaudio.PyAudio()

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

            print("recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")


            # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # saving wav file 
            waveFile = wave.open(FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            # load model 
            model = pickle.load(open(MODEL,'rb'))

             # reading audio files of speaker
            (sr, audio) = read(FILENAME)
            
            # extract 40 dimensional MFCC
            vector = extract_features(audio,sr)
            vector = vector.flatten()
            test_audio = vector

            # predict with model
            pred = model.predict(test_audio.reshape(1,-1))

            # decode predictions
            le = preprocessing.LabelEncoder()
            le.fit(VOICENAMES)
            identity = le.inverse_transform(pred)[0]

            # if voice not recognized than terminate the process
            if identity == 'unknown':
                    print("Not Recognized! Try again...")
                    time.sleep(1.5)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
            
            print( "Recognized as - ", identity)
            return
            
    except KeyboardInterrupt:
        print("Stopped")
        pass
    
if __name__ == '__main__':
    recognize()
