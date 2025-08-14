# 💤 Yawning Detection System

An **AI-powered real-time yawning detection system** built using **OpenCV** and **MediaPipe** to monitor drivers for signs of fatigue.  
If the user yawns **3 or more times within 1 minute**, the system will **trigger a buzzer and/or voice alert** to help prevent drowsy driving accidents.

---

## 🚀 Features
- Real-time yawning detection using webcam
- AI-based facial landmark tracking (MediaPipe)
- Adjustable sensitivity thresholds
- **Triggers buzzer if 3+ yawns occur in under 60 seconds**
- FPS display for performance monitoring

---

## 🛠️ Technologies Used
- **Python 3.x**
- **OpenCV** – Video capture & image processing
- **MediaPipe** – Facial landmark detection
- **pyttsx3 / playsound** – Voice and buzzer alerts

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhargav-G17/Yawning.git
   cd Yawning
