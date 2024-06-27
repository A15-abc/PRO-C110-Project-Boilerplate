import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame to the size expected by the model, e.g., 224x224
        resized_frame = cv2.resize(frame, (224, 224))

        # Expand the dimensions to match the input shape of the model
        expanded_frame = np.expand_dims(resized_frame, axis=0)

        # Normalize the frame (assuming the model expects values between 0 and 1)
        normalized_frame = expanded_frame / 255.0

        # Get predictions from the model
        predictions = model.predict(normalized_frame)

        # Assuming the model returns a list of predictions
        predicted_class = np.argmax(predictions, axis=1)

        # Display the prediction on the frame
        label_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
        prediction_text = label_map[predicted_class[0]]
        cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)
        
        # if space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
