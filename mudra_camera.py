import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image

from mudra_model import CNN
from gesture_names import extract_gesture_names

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('mudra_model.pth'))
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # Resize images to 224x224
    transforms.Grayscale(num_output_channels=1),    # Convert images to grayscale
    transforms.ToTensor(),                          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485], std=[0.229]) # Normalize images (single channel)
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
     # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    frame_pil = Image.fromarray(frame_rgb)

    # Apply transformations
    input_tensor = transform(frame_pil)
    input_batch = input_tensor.unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        predictions = model(input_batch)
        _, predicted_idx = torch.max(predictions, 1)

    # Get the gesture names
    gesture_names = extract_gesture_names()

    # Map the predicted index to the corresponding gesture name
    predicted_idx = predicted_idx.item()
    predicted_gesture = gesture_names[predicted_idx]

    # Display the resulting frame
    cv2.putText(frame, f'Predicted Gesture: {predicted_gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
