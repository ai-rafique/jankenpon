import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('weightsrps.hdf5')
move = ['paper', 'rock', 'scissor']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # rectangle for user to play
    cv2.rectangle(frame, (0, 0), (250, 250), (0, 0, 255), 2)
    cv2.rectangle(frame, (389, 0), (639, 250), (255, 0, 0), 2)

    # Player 1
    roi = frame[0:250, 0:250]

    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    test_image = np.expand_dims(img, axis=0)
    result = model.predict(test_image)
    i = np.argmax(result)

    # Player 2
    roi2 = frame[0:250, 389:639]

    img2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (150, 150))

    test_image2 = np.expand_dims(img2, axis=0)
    result2 = model.predict(test_image2)
    i2 = np.argmax(result2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, move[i], (50, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, move[i2], (430, 50), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    # Tie case, P1 case, else P2
    if i2 == i:
        cv2.putText(frame, 'Its a tie', (250, 350), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    elif i == 0 and i2 == 1:
        cv2.putText(frame, 'P1 Wins', (250, 350), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    elif i == 1 and i2 == 2:
        cv2.putText(frame, 'P1 Wins', (250, 350), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    elif i == 2 and i2 == 0:
        cv2.putText(frame, 'P1 Wins', (250, 350), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'P2 Wins', (250, 350), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
