## Face Recognition

Modern face recognition with deep learning and HOG algorithm.  

1. Find faces in image (HOG Algorithm)   
2. Affine Transformations (Face alignment using an ensemble of regression
trees)   
3. Encoding Faces (FaceNet)  
4. Make a prediction (Linear SVM)  

We are using the [Histogram of Oriented Gradients](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (HOG) method. Instead of computing gradients for every pixel of the image (way too much detail). We compute the weighted vote orientation  gradients of 16x16 pixels squares. Afterward, we have a simple representation (HOG image) that captures the basic structure of a face.  
All we have to do is find the part of our image that looks the most similar to a known trained HOG pattern.  
For this technique, we use the dlib Python library to generate and view HOG representations of images.  
```
face_detector = dlib.get_frontal_face_detector()
detected_faces = face_detector(image, 1)
```

After isolating the faces in our image, we need to warp (posing and projecting)the picture so the face is always in he same place. To do this, we are going to use the [face landmark estimation algorithm](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf). Following this method, there are 68 specific points (landmarks) on every face and we train a machine learning algorithm to find these 68 specific points on any face. 
```
face_pose_predictor = dlib.shape_predictor(predictor_model)
pose_landmarks = face_pose_predictor(img, f)
```
After find those landmarks, we need to use affine transformations (such as rotating, scaling and shearing --like translations) on the image so that the eyes and mouth are centered as best as possible.
```
face_aligner = openface.AlignDlib(predictor_model)
alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
```

The next step is encoding the detected face. For this, we use Deep Learning. We train a neural net to generate [128 measurements (face embedding)](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) for each face.  
