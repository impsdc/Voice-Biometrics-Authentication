# Voice Biometrics Authentication<br>

Voice Biometrics Authentication using GMM

---

## How to Run :

**Create virtualenv** `python3 -m venv venv && source venv/bin/activate`<br>
**Install dependencies by running** `pip3 install -r requirement.txt`<br>

### 2.Before running :

- Test if your microphone is listenable
- Make sure you are in a quit place

### 3.Run in terminal in following way :

**To add new user :**

```
  python3 add_user.py
```

**To Recognize user :**

```
  python3 recognize.py
```

**To Recognize until KeyboardInterrupt (ctrl + c) by the user:**

```
  python3 recognize_until_keyboard_Interrupt.py
```

**To delete an existing user :**

```
  python3 delete_user.py
```

---

## In case recognition is not working

Make sure that when your recognize user, you are in the same place where your registred him.<br>
Use the same microphone you registred with.

## Voice Authentication

_For Voice recognition, **GMM (Gaussian Mixture Model)** is used to train on extracted MFCC features from audio wav file._<br><br>
