import cv2
import mediapipe as mp
import math
import pyautogui
import threading
import time

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe and configure MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        #Acuracy precision and detection of handse
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            static_image_mode=False, #if you user a video and no camera use True
            max_num_hands=1, #Max number of hands
              min_tracking_confidence=0.5)
        
        # State variables
        self.space_pressed = False
        self.lock = threading.Lock()
        
        # Constants
        self.PINCH_THRESHOLD = 60  # Distance threshold for pinch detection
        
    def press_space(self):
        #Press downspace key
        with self.lock:
            pyautogui.keyDown('space')
            print("Space key pressed")
            
    def release_space(self):
        #Up space key
        with self.lock:
            pyautogui.keyUp('space')
            print("Space key released")
            
    def calculate_distance(self, point1, point2, frame_width, frame_height):
        x1, y1 = int(point1.x * frame_width), int(point1.y * frame_height)
        x2, y2 = int(point2.x * frame_width), int(point2.y * frame_height)
        return math.hypot(x1 - x2, y1 - y2)
    
    def process_frame(self, frame):
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Draw hand landmarks and check for gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get landmark positions
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                
                # Calculate distance between index finger and thumb
                distance = self.calculate_distance(index_finger_tip, thumb_tip, w, h)
                
                # Display distance on frame
                cv2.putText(frame, f"Distance: {int(distance)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Check for pinch gesture
                if distance > self.PINCH_THRESHOLD and not self.space_pressed:
                    threading.Thread(target=self.press_space).start()
                    self.space_pressed = True
                elif distance <= self.PINCH_THRESHOLD and self.space_pressed:
                    threading.Thread(target=self.release_space).start()
                    self.space_pressed = False
        
        return frame
    
    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow("Hand Gesture Space Key Control", processed_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
            # Make sure space key is released
            if self.space_pressed:
                pyautogui.keyUp('space')

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()