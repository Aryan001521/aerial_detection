# 🦅 Bird vs Drone Detection System

An AI-powered web application that detects and classifies **Birds 🐦 and Drones 🚁** from images using **YOLOv8 Object Detection** and optional **CNN Classification**.

---

## 🚀 Features

✅ Real-time Object Detection using YOLOv8  
✅ Detect multiple objects in a single image  
✅ Bounding Box + Confidence Score  
✅ Modern Streamlit UI  
✅ Optional CNN Classification (ResNet50)  
✅ Clean and interactive dashboard  

---

## 🧠 Model Details

### 🔹 YOLOv8 (Detection)
- Custom trained model
- Classes: `Bird`, `Drone`
- Fast inference (~7ms per image)
- File: `best.pt`

### 🔹 CNN (Optional Classification)
- Model: ResNet50
- Output: Bird / Drone classification
- File: `best_resnet50.pth`

---

## 📁 Project Structure
