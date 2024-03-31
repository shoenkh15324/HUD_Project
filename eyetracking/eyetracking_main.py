import cv2
import dlib
import pyautogui

# 얼굴 검출기와 눈 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 랜드마크를 표시하는 함수 정의
def draw_face_landmarks(frame, face):
          landmarks = predictor(gray, face)
          for i in range(68):
               x, y = landmarks.part(i).x, landmarks.part(i).y
               cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
               
# 양쪽 눈의 중심을 찾는 함수 정의
def find_eye_centers(face):
     landmarks = predictor(gray, face)
     left_eye = landmarks.part(36).x, landmarks.part(37).y, landmarks.part(39).x - landmarks.part(36).x, landmarks.part(41).y - landmarks.part(37).y
     right_eye = landmarks.part(42).x, landmarks.part(43).y, landmarks.part(45).x - landmarks.part(42).x, landmarks.part(47).y - landmarks.part(43).y
     return left_eye, right_eye

# 눈 중심을 표시하는 함수 정의
def draw_eye_centers(frame, eyes):
     x1, y1, w1, h1 = eyes[0]
     eye1_center_x = x1 + w1 // 2
     eye1_center_y = y1 + h1 // 2
          
     x2, y2, w2, h2 = eyes[1]
     eye2_center_x = x2 + w2 // 2
     eye2_center_y = y2 + h2 // 2
     
     cv2.circle(frame, (eye1_center_x, eye1_center_y), 2, (0, 255, 0), -1)
     cv2.putText(frame, f"({eye1_center_x}, {eye1_center_y})", (eye1_center_x - 35, eye1_center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
     
     cv2.circle(frame, (eye2_center_x, eye2_center_y), 2, (0, 255, 0), -1)
     cv2.putText(frame, f"({eye2_center_x}, {eye2_center_y})", (eye2_center_x - 35, eye2_center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
     
     print("left_eye:",eyes[0],", right_eye:",eyes[1])

# 시선을 표시하는 함수 정의
def draw_eye_sight(frame, eyes):
     x1, y1, w1, h1 = eyes[0]
     eye1_center_x = x1 + w1 // 2
     eye1_center_y = y1 + h1 // 2
          
     x2, y2, w2, h2 = eyes[1]
     eye2_center_x = x2 + w2 // 2
     eye2_center_y = y2 + h2 // 2
          
     # 두 눈 중심의 중앙값으로 시선 좌표 계산
     eye_sight_x = (eye1_center_x + eye2_center_x) // 2
     eye_sight_y = (eye1_center_y + eye2_center_y) // 2
          
     cv2.circle(frame, (eye_sight_x, eye_sight_y), 2, (255, 0, 0), -1)
     print("eye sight:",(eye_sight_x, eye_sight_y))
     cv2.putText(frame, f"({eye_sight_x}, {eye_sight_y})", (eye_sight_x - 35, eye_sight_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
     
# 시선을 따라가는 윈도우창을 구현하는 함수 정의
def follow_gaze_window(frame, eyes, sensitivity):
     x1, y1, w1, h1 = eyes[0]
     eye1_center_x = x1 + w1 // 2
     eye1_center_y = y1 + h1 // 2
     
     x2, y2, w2, h2 = eyes[1]
     eye2_center_x = x2 + w2 // 2
     eye2_center_y = y2 + h2 // 2
     
     eye_sight_x = (eye1_center_x + eye2_center_x) // 2
     eye_sight_y = (eye1_center_y + eye2_center_y) // 2
     
     Gaze_Follower_width = 360
     Gaze_Follower_height = 360
     
     # 시선 반응에 대한 감도 조절
     eye_sight_x_offset = int(sensitivity * (eye_sight_x - frame.shape[1] // 2))
     eye_sight_y_offset = int(sensitivity * (eye_sight_y - frame.shape[0] // 2))
     
     # 현재 화면 해상도 가져오기
     screen_width, screen_height = pyautogui.size()
     
     # 새로운 윈도우 위치 계산
     new_window_x = screen_width // 2 + eye_sight_x_offset
     new_window_y = screen_height // 2 + eye_sight_y_offset
     
     # 윈도우가 프레임 경계를 벗어나지 않도록 보정
     if new_window_x < 0:
          new_window_x = 0
     elif new_window_x + Gaze_Follower_width > screen_width:
          new_window_x = screen_width - Gaze_Follower_width
          
     if new_window_y < 0:
          new_window_y = 0
     elif new_window_y + Gaze_Follower_height > screen_height:
          new_window_y = screen_height - Gaze_Follower_height
     
     # 윈도우 이동
     cv2.namedWindow('Gaze Follower')
     cv2.resizeWindow('Gaze Follower', Gaze_Follower_width, Gaze_Follower_height)
     cv2.moveWindow('Gaze Follower', new_window_x, new_window_y)


# 웹캠 시작
cap = cv2.VideoCapture(0)

while True:
     # 프레임 읽기
     ret, frame = cap.read()
     if not ret:
          break
     
     # 화면 뒤집기
     frame = cv2.flip(frame, 1)

     # 그레이스케일 변환
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # 얼굴 검출
     faces = detector(gray)

     for face in faces:
          eyes = find_eye_centers(face) # 양쪽 눈의 중심 좌표 구하기
          draw_face_landmarks(frame, face)
          draw_eye_centers(frame, eyes) # 눈 좌표 그리기
          draw_eye_sight(frame, eyes) # 시선 좌표 그리기
          follow_gaze_window(frame, eyes, 10.0)
          
     
     # 결과 출력
     cv2.imshow('Eye Tracker', frame)

     # 'q'를 누르면 종료
     if cv2.waitKey(1) == 27:
          break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
