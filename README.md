# Hardware-Project---Chess-Navigator
**ChessNavigator** is an automated chess-playing system that combines **computer vision**, **robotics**.  
It detects moves in real time using a Raspberry Pi camera, processes them with OpenCV, and plays using a robotic arm controlled via Arduino and a custom PCB.

## 🚀 Features
- 🎥 **Vision System** – Real-time chessboard and move detection using OpenCV.  
- 🤖 **Robotic Arm** – Servo-controlled gripper to pick and place chess pieces.  
- 🧠 **AI Engine** – Play against difficulty levels (Easy, Medium, Hard).  
- 💡 **Lighting & Calibration** – Stable move detection under varying light.  
- 🔌 **Custom PCB** – Integrated Arduino Nano, servo driver, and power system.  

## 📂 Project Structure

- ChessNavigator/
  -   docs/                # Documentation, reports, diagrams, presentations
    -  images/          # Screenshots and demo pictures
    - hardware/            # PCB design files, schematics, calibration data
   -  microcontroller/     # Arduino code for servo & robotic arm control
       -  servo_control.ino
    - vision/              # Python OpenCV chess detection system
      -   chess_tracker.py
       -  calibration.py
  -   ui/                  # Pygame UI code for chess interface
    -  chess_main.py
    - engine/              # Chess engine + smart move finder
      -   smartMoveFinder.py
        - chessengine.py
    - tests/               # Testing scripts for system components


## ⚙️ Tech Stack
- **Python** (OpenCV, NumPy, Flask)  
- **Arduino (C++)** with PCA9685 servo driver  
- **Raspberry Pi** for vision & processing  
- **Custom PCB** for servo + power integration  


👥 Team

- **Poorna Danushka** (234095F)
- **Lisitha** (234115U)
- **Chathuni** (234098R)
- **Chamalka** (234079K)
- **Pavidu** (234127H)
