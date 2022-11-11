import os 
import pickle
import glob
import wave
import cv2
from main_functions import *

# deletes a registered user from database
def delete_user():
    VOICEPATH = "voice_database/"

    name = input("Enter name of the user:")   

    users = [ name for name in os.listdir(VOICEPATH) if os.path.isdir(os.path.join(VOICEPATH, name)) ]
    
    if name not in users or name == "unknown":
        print('No such user !!')

    [os.remove(path) for path in glob.glob(VOICEPATH + name + '/*')]
    os.removedirs(VOICEPATH + name)

    if os.path.isfile("gmm_models/voice_auth.gmm"): 
        os.remove("gmm_models/voice_auth.gmm")
        
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

    print('User ' + name + ' deleted successfully')


if __name__ == '__main__':
    delete_user()