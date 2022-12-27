import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
import joblib

cap = cv2.VideoCapture(0)

#removes unneccessary data
def data_clean(landmark):
  
  data = landmark[0]
  
  try:
    data = str(data)

    data = data.strip().split('\n')

    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

    without_garbage = []

    for i in data:
        if i not in garbage:
            without_garbage.append(i)

    clean = []

    for i in without_garbage:
        i = i.strip()
        clean.append(i[2:])

    for i in range(0, len(clean)):
        clean[i] = float(clean[i])

    
    return([clean])

  except:
    return(np.zeros([1,63], dtype=int)[0])


#configures mediapipe hands
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  #opening video feed
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #processes image with mp hands
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #draws hand landmarks onto image
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        #cleans landmark
        cleaned_landmark = data_clean(results.multi_hand_landmarks)
        
        if cleaned_landmark:
          #calling model
          model = joblib.load('ISL_model.pkl')
          #making prediction
          y_pred = model.predict(cleaned_landmark)
          #displays prediction on screen
          #image = cv2.putText(image, text, bottomleftcoord(x,y), cv2.font, fontScale, textcolor(b,g,r), thickness(px), cv.linetype)
          image = cv2.putText(image, str(y_pred[0]), (50,150), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,128,0), 3, cv2.LINE_AA)

    #shows image
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    # if escape key exits breaks loop
    if cv2.waitKey(5) & 0xFF == 27:
      break

#closes
hands.close()
cap.release()
cv2.destroyAllWindows()