import cv2 

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
# Trained tada by OPENCV (machine learning)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('rdj.png')

# To capture video from webcam
webcam = cv2.VideoCapture(1)

# Iterate forever over frames
while True:

  # Read the current frame
  successful_frame_read, frame = webcam.read()
  # print("Successful frame read:", successful_frame_read)

  # Must convert to grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


  # Detect faces
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

  # Draw rectangles around the faces
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  cv2.imshow('Clever Programmer Face Detector', frame)
  key = cv2.waitKey(1)

  if key==81 or key==113:
    break

  # Release the webcam (video capture object)
webcam.release()

print("Code Completed")

  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #     break


# Release the webcam
# webcam.release()

# Close all OpenCV windows
# cv2.destroyAllWindows()



# # Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# # Draw rectangles around the faces
# for (x, y, w, h) in face_coordinates:
#   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # print(face_coordinates)
# # Display the image with the faces
# cv2.imshow('Clever Programmer Detector', img)
# cv2.waitKey()

# print("Code Completed")
