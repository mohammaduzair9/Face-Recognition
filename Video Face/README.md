# Face-Recognition

This is a facial recognition system on video.

## Implementation

The project is implemented in 3 phases:
  1. Face Detection is done using MTCNN.
  2. Face Alignmnet is done using MTCNN.
  3. Face Recognition is done using FaceNet.
  
## How to run
 
 Place video file in project directory.
 Go to project directory and run the face_video.py stored in src folder.<br />
    'python src/face_video.py'<br />
    
 This is only detection.
 For recognition pass bounding boxes and get prediction. (To be implemented)

## Credits
 
 The code is mainly taken from https://github.com/davidsandberg/facenet
