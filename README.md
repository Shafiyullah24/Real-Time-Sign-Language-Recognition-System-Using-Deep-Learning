# Real-Time-Sign-Language-Recognition-System-Using-Deep-Learning
This project presents a real-time sign language recognition system that translates hand gestures into readable text using computer vision and deep learning techniques. The system utilizes a webcam to capture live video input and detects hand landmarks using Mediapipe.
The system uses a trained Convolutional Neural Network (CNN) model to classify hand signs and displays the recognized letters on screen, forming words and sentences.

🚀 Features
  📷 Real-time webcam-based gesture detection
  🧠 Deep learning model for accurate classification
  ✋ Hand landmark detection using Mediapipe
  🔤 Converts gestures into text (sentence formation)
  📊 Confidence score and top predictions display
  ⏳ Gesture confirmation using time-based stability
  🧹 Clear sentence option

🗂️ Project Structure
  📁 dataset_custom/        # Collected dataset for training
  📄 custom_model_capture.py # Script to capture and save gesture images
  📄 test_train_save.py      # Model training and saving script
  📄 Sign_recognition(main).py # Main real-time detection system
  📄 README.md               # Project documentation
  
⚙️ Requirements
Install the required libraries using:
  pip install opencv-python numpy tensorflow mediapipe

🧠 How It Works
  1) Data Collection
    Hand gesture images are captured using custom_model_capture.py
    Images are stored in dataset_custom
  2) Model Training
    test_train_save.py trains a CNN model on collected data
    Model is saved as .keras file
    Class labels are stored in .npy file
  3) Real-Time Recognition
    Sign_recognition(main).py uses webcam input
    Mediapipe detects hand landmarks
    Region of interest (ROI) is extracted and passed to the model
    Predicted character is displayed
  4) Sentence Formation
    A character is confirmed only if held steady for a few seconds
    Characters are appended to form a sentence.

