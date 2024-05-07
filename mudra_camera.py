import cv2
import torch
import torchvision.transforms as transforms
import mediapipe as mp
from PIL import Image
from torchvision import models
from gesture_names import extract_gesture_names

# Initialize the Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the Mediapipe Drawing module
mp_drawing = mp.solutions.drawing_utils

# Load the pretrained ResNet model and modify it for your gesture classification
model = models.resnet18(pretrained=False)
num_classes = 51
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Load the trained weights from the ResNet model file
model.load_state_dict(torch.load('mudra_model_resnet18.pth'))
model.eval()

# Define preprocessing transformations consistent with the ResNet model
transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # Resize to 224x224
    transforms.ToTensor(),                          # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use ImageNet stats
])

# Function to calculate the bounding box from landmarks
def get_bounding_box(landmarks, image_width, image_height):
    x_coords = [landmark.x * image_width for landmark in landmarks]
    y_coords = [landmark.y * image_height for landmark in landmarks]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return x_min, y_min, x_max, y_max

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # Initialize the predicted gesture as "No Hand Detected"
    predicted_gesture = "No Hand Detected"

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box coordinates for the detected hand
            x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks.landmark, frame.shape[1], frame.shape[0])

            # Crop the hand region from the frame
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Convert the cropped hand region to a PIL Image and apply transformations
            hand_pil = Image.fromarray(hand_roi)
            input_tensor = transform(hand_pil)
            input_batch = input_tensor.unsqueeze(0)

            # Make a prediction with the ResNet model
            with torch.no_grad():
                predictions = model(input_batch)
                _, predicted_idx = torch.max(predictions, 1)

            # Get the gesture names
            gesture_names = extract_gesture_names()

            # Map the predicted index to the corresponding gesture name
            predicted_idx = predicted_idx.item()
            predicted_gesture = gesture_names[predicted_idx]

    # Display the predicted gesture on the frame
    cv2.putText(frame, f'Predicted Gesture: {predicted_gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
