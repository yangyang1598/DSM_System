## pip install cmake
## pip install dlib
## pip install keras
## pip install tensorflow
## pip install imutils

from MyLibrary import myopencv as my
import dlib, cv2, copy
import numpy as np
from keras.models import load_model
from imutils import face_utils

def crop_eye(img, IMG_SIZE, eye_points):
  x1, y1 = np.amin(eye_points, axis=0) ## 눈 좌표의 왼쪽 부분 위치 추출
  x2, y2 = np.amax(eye_points, axis=0) ## 눈 좌표의 오른쪽 부분 위치 추출
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 ## 눈 좌표의 센터 추출

  w = (x2 - x1) * 1.2 ## 눈 좌표를 이용하여 눈 이미지 weigth 지정
  h = w * IMG_SIZE[1] / IMG_SIZE[0] ## 눈 이미지 weigth 비율에 맞춰 / 눈 이미지 high

  margin_x, margin_y = w / 2, h / 2 ## weigth , high 중앙

  min_x, min_y = int(cx - margin_x), int(cy - margin_y) ## 조금 더넓게 이미지를 가져오기 위함
  max_x, max_y = int(cx + margin_x), int(cy + margin_y) ## 조금 더넓게 이미지를 가져오기 위함

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int) ## 소수를 반올림 하여 정수형으로 저장

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]] ## 눈이미지만 따옴

  return eye_img, eye_rect

model = load_model('models/eyes_blink_detector.h5')
model.summary()

Dir_Path_Data = 'dataset/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

Color = (0, 255, 0)
IMG_SIZE = (34, 26)

Video = cv2.VideoCapture('video/4.mp4')
Out = cv2.VideoWriter('eyes_blink_detector.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 30, (1216, 1080))
count = 0
frame = 0
wakeup = 0
Faces = []
frametosec = 30

while True:
    Text = []
    frame += 1
    _, img = Video.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) >= 1:
        Faces.append(detector(gray))
        faces_temp = Faces[-1]
    else:
        faces_temp = Faces[-1]

    img_r = img.copy()
    img_s = img.copy()
    eye_r = img.copy()

    for face in faces_temp:
        p1 = (face.left(), face.top())
        p2 = (face.right(), face.bottom())

        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, IMG_SIZE, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, IMG_SIZE, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        input_l = eye_img_l.reshape(1, IMG_SIZE[1], IMG_SIZE[0], 1)
        input_r = eye_img_r.reshape(1, IMG_SIZE[1], IMG_SIZE[0], 1)

        pred_l = model.predict(input_l)
        print(pred_l)
        pred_r = model.predict(input_r)

        if len(faces) == 0:
            cv2.rectangle(eye_r, p1, p2, color=(0, 0, 255), thickness=2)
        else:
            cv2.rectangle(eye_r, p1, p2, color=Color, thickness=2)

        color_l = (255, 255, 0)
        color_r = (0, 255, 255)

        for num, (x, y) in enumerate(shapes):
            center = (x, y)
            if num >= 36 and num < 42:
                cv2.circle(eye_r, center, 2, color=color_l, thickness=-1)
            elif num >= 42 and num < 48:
                cv2.circle(eye_r, center, 2, color=color_r, thickness=-1)
            else:
                cv2.circle(eye_r, center, 2, color=(255, 255, 255), thickness=-1)

        if pred_l >= 0.5:
            Color_l = (0, 255, 0)
        else:
            Color_l = (0, 0, 255)

        if pred_r >= 0.5:
            Color_r = (0, 255, 0)
        else:
            Color_r = (0, 0, 255)

        if pred_l < 0.5 or pred_r < 0.5:
            count += 1
        elif pred_l >= 0.5 and pred_r >= 0.5 and wakeup >= 30:
            count = 0

        if pred_l >= 0.5 and pred_r >= 0.5 and count > 0:
            wakeup += 1

        if len(faces) == 0:
            cv2.putText(img, f'Sleep : {count / frametosec:.2f}sec', (p1[0] + 10, p1[1] - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        elif int(count / frametosec) >= 2 and int(count / frametosec) < 4:
            cv2.putText(img, f'Boring : {count / frametosec:.2f}sec', (p1[0] + 10, p1[1] - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        elif int(count / frametosec) >= 4 and int(count / frametosec) < 6 :
            cv2.putText(img, f'Waring : {count / frametosec:.2f}sec', (p1[0] + 10, p1[1] - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        elif int(count / frametosec) >= 6:
            cv2.putText(img, f'Sleep : {count / frametosec:.2f}sec', (p1[0] + 10, p1[1] - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(img, f'Nomal : {count / frametosec:.2f}sec', (p1[0] + 10, p1[1] - 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
            pass

        cv2.rectangle(eye_r, tuple(eye_rect_l[0:2]), tuple(eye_rect_l[2:]), color=Color_l, thickness=2)
        cv2.rectangle(eye_r, tuple(eye_rect_r[0:2]), tuple(eye_rect_r[2:]), color=Color_r, thickness=2)

    h, w = eye_r.shape[:2]
    if len(faces) == 0:
        Text.append('Faces_Detecting_Error')
    else:
        Text.append('Faces_Detecting')
        pass
    if len(pred_l) == 0 and len(pred_r) == 0:
        Text.append('Eyes_UnDetecting')
    else:
        Text.append('Eyes_Detecting')
        pass
    cv2.putText(img, f'{frame}', (0, h - 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f'{Text[0] + " / " + Text[1]}', (0, 15), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

    face_img = img[p1[1]:p2[1], p1[0]:p2[0]]
    eye_l_img = img[eye_rect_l[1]:eye_rect_l[3], eye_rect_l[0]:eye_rect_l[2]]
    eye_r_img = img[eye_rect_r[1]:eye_rect_r[3], eye_rect_r[0]:eye_rect_r[2]]

    scale = 203 / face_img[1]

    face_img = cv2.resize(face_img, dsize=(304, 200))
    eye_l_img = cv2.resize(eye_l_img, dsize=(304, 100))
    eye_r_img = cv2.resize(eye_r_img, dsize=(304, 100))

    step_1 = cv2.vconcat([eye_l_img, eye_r_img])
    step_2 = cv2.hconcat([face_img, step_1])
    step_3 = cv2.vconcat([step_2, eye_r])

    step_3 = cv2.resize(step_3, dsize=(608, 1080))

    result = cv2.hconcat([img, step_3])

    print(result.shape)
    Out.write(result)
    my.imgshow('result', result, 1000)

    key = cv2.waitKey(1)
    if key == 114:
        cv2.waitKey(0)
    if key == 27:
        Out.release()
        Video.release()
        cv2.destroyAllWindows()
        break

Out.release()
Video.release()
cv2.destroyAllWindows()