import cv2
import dlib

# 얼굴 검출기와 눈 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이전 프레임에서 눈이 감겼는지 여부를 추적하는 변수 초기화
prev_left_eye_closed = False
prev_right_eye_closed = False

# 눈 깜빡임 횟수를 저장하는 변수 초기화
left_blink_count = 0
right_blink_count = 0

# 눈 중심을 표시하는 함수 정의
def draw_eye_centers(frame, eyes):
    for (x, y, w, h) in eyes:
        eye_center_x = x + w // 2
        eye_center_y = y + h // 2
        cv2.circle(frame, (eye_center_x, eye_center_y), 2, (0, 255, 0), -1)
        cv2.putText(frame, f"({eye_center_x}, {eye_center_y})", (eye_center_x + 5, eye_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
          # 얼굴 landmarks 표시
          landmarks = predictor(gray, face)
          for i in range(68):
               x, y = landmarks.part(i).x, landmarks.part(i).y
               cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
          # 눈 검출
          landmarks = predictor(gray, face)
          left_eye = landmarks.part(36).x, landmarks.part(37).y, landmarks.part(39).x - landmarks.part(36).x, landmarks.part(41).y - landmarks.part(37).y
          right_eye = landmarks.part(42).x, landmarks.part(43).y, landmarks.part(45).x - landmarks.part(42).x, landmarks.part(47).y - landmarks.part(43).y

          draw_eye_centers(frame, [left_eye, right_eye])
  
     # 결과 출력
     cv2.imshow('Eye Tracker', frame)

     # 'q'를 누르면 종료
     if cv2.waitKey(1) == 27:
          break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
