import tkinter as tk
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Hardcoded paths to the prototxt and model files
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Global variable to track last spoken time
last_spoken_time = 0
speak_delay = 4  # Delay in seconds for speech

# Function to start the video detection
def start_detection():
    global last_spoken_time
    
    # Initialize video stream and FPS counter
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    fps = FPS().start()

    # Loop over the frames from the video stream
    while True:
        # Grab the frame and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # Blob creation for passing through the model
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Confidence threshold
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw bounding box and label on frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Check the delay before speaking the label
                current_time = time.time()
                if current_time - last_spoken_time > speak_delay:
                    engine.say(CLASSES[idx])
                    engine.runAndWait()
                    last_spoken_time = current_time

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break loop on 'q' key press
        if key == ord("q"):
            break

        # Update FPS counter
        fps.update()

    # Stop FPS counter and display info
    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()

# Setting up GUI using tkinter
window = tk.Tk()
window.title("Object Detection with Speech Output")
window.geometry("400x300")

# Project title label
title_label = tk.Label(window, text="Object Detection with Speech", font=("Helvetica", 16), fg="#4682B4")
title_label.pack(pady=20)

# Start button to initiate detection
start_button = tk.Button(window, text="Start Detection", font=("Helvetica", 14), command=start_detection, bg="#4682B4", fg="#FFFFFF")
start_button.pack(pady=20)

# Instructions label
instructions = tk.Label(window, text="Press 'q' to quit the video detection", font=("Helvetica", 10))
instructions.pack(pady=10)

# Start the GUI loop
window.mainloop()
