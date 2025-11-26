# 🎤 Speech Emotion Recognition (SER)

This project detects human emotions from speech audio using MFCC features and a trained Deep Learning model.  
It is made as part of the ADS (Advanced Data Structures) project.

---

## 👩‍💻 Project Details
**Project Title:** Speech Emotion Recognition  
**Made By:** Laviza  
**Subject:** ADS – Advanced Data Structures  
**GitHub Repository:** https://github.com/LAVIZA123/speech-emotion-recognition

---

## 📌 Objective
The main objective of this project is to classify emotions from speech such as:

- Happy  
- Sad  
- Angry  
- Neutral  
- Fear  
- Surprise  

By extracting audio patterns using MFCC and passing them to a trained model.

---

## 🧠 How the System Works
1. User provides a `.wav` audio file  
2. MFCC features are extracted  
3. Features are given to the trained model  
4. Model returns the predicted emotion  

---

## 🏗️ Project Structure
```plaintext
speech-emotion-recognition/
│
├── src/
│   ├── extract_features.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   ├── model_architecture.json
│   └── model_weights.weights.h5
│
├── samples/
├── requirements.txt
└── README.md
```

---

## 🎼 Feature Extraction (MFCC)
We convert audio into numerical features using MFCC.

```python
mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
mfcc_scaled = np.mean(mfcc.T, axis=0)
```

---

## 🤖 Model Used
This project uses a **Convolutional Neural Network (CNN)** for emotion classification.

Saved files:
- model.pkl – Trained model  
- model_architecture.json – CNN architecture  
- model_weights.weights.h5 – Model weights  

---

## ⚙️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```bash
python src/train.py
```

### 3️⃣ Predict Emotion from Audio
```bash
python src/predict.py --file samples/example.wav
```

### Sample Output
```
Predicted Emotion: Happy
Confidence: 92%
```

---

## 📊 Expected Output
- Detects emotion from audio  
- Shows confidence score  
- Works on `.wav` files  

---

## 🚀 Future Enhancements
- Real-time emotion detection  
- Live microphone recording  
- GUI or web app  
- Larger dataset for improved accuracy  

---

## 🤝 Contribution
You can improve the project by making a pull request.

---

## 📞 Contact
GitHub Profile:  
https://github.com/LAVIZA123

---

## 📜 License
This project is licensed under the MIT License.
