import cv2
import dlib
import pyautogui
import numpy as np

# 화면에 표시하는 기능과 관련된 클래스
class EyeTracker:
     def __init__(self):
          self.detector = dlib.get_frontal_face_detector()
          self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

     def draw_face_landmarks(self, frame, face):
          """
          Draw landmarks on the face.

          Args:
               frame (numpy.ndarray): Image frame.
               face (dlib.rectangle): Detected face.

          Returns:
               None
          """
          landmarks = self.predictor(frame, face)
          for i in range(68):
               x, y = landmarks.part(i).x, landmarks.part(i).y
               cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

     def draw_eye_centers(self, frame, eyes):
          """
          Draw the centers of both eyes.

          Args:
               frame (numpy.ndarray): Image frame.
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.

          Returns:
               None
          """
          coordinates = EyeCoordinate(eye_tracker)
          centers = coordinates.get_centers(eyes)

          for center in centers:
               cv2.circle(frame, (center[0], center[1]), 2, (0, 255, 0), -1)
               cv2.putText(frame, f"({center[0]}, {center[1]})", (center[0] - 35, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

          # print("left_eye:", centers[0], ", right_eye:", centers[1])

     def draw_eye_sight(self, frame, eyes):
          """
          Draw the estimated eye sight.

          Args:
               frame (numpy.ndarray): Image frame.
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.

          Returns:
               None
          """
          coordinate = EyeCoordinate(eye_tracker)
          sight = coordinate.get_sight(frame, eyes)

          cv2.circle(frame, (sight[0], sight[1]), 2, (255, 0, 0), -1)
          # print("eye sight:", sight)
          cv2.putText(frame, f"({sight[0]}, {sight[1]})", (sight[0] - 35, sight[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
          
     def detect_and_draw_pupils(self, frame, eyes):
          """
          Detect pupils in the eye regions and draw them as circles on the frame.

          Args:
               frame (numpy.ndarray): Image frame.
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.

          Returns:
               None
          """
          for eye in eyes:
               pupil_x, pupil_y, radius = pupil_detector.detect_pupil_using_black_detection(frame, eye)
               cv2.circle(frame, (pupil_x, pupil_y), radius, (0, 255, 255), 0)
               cv2.putText(frame, f"({pupil_x}, {pupil_y})", (pupil_x - 5, pupil_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
          
          # Print coordinates of left and right pupils
          left_pupil_x, left_pupil_y, _ = pupil_detector.detect_pupil_using_black_detection(frame, eyes[0])
          right_pupil_x, right_pupil_y, _ = pupil_detector.detect_pupil_using_black_detection(frame, eyes[1])
          print("Left pupil:", (left_pupil_x, left_pupil_y), "Right pupil:", (right_pupil_x, right_pupil_y))

     def follow_gaze_window(self, frame, eyes, sensitivity):
          """
          Implement a window that follows the estimated eye gaze.

          Args:
               frame (numpy.ndarray): Image frame.
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.
               sensitivity (float): Sensitivity factor for eye gaze.

          Returns:
               None
          """
          coordinate = EyeCoordinate(eye_tracker)
          eye_sight = coordinate.get_sight(frame, eyes)

          gaze_follower_width = 360
          gaze_follower_height = 360

          eye_sight_x_offset = int(sensitivity * (eye_sight[0] - frame.shape[1] // 2))
          eye_sight_y_offset = int(sensitivity * (eye_sight[1] - frame.shape[0] // 2))

          screen_width, screen_height = pyautogui.size()

          new_window_x = screen_width // 2 + eye_sight_x_offset
          new_window_y = screen_height // 2 + eye_sight_y_offset

          if new_window_x < 0:
               new_window_x = 0
          elif new_window_x + gaze_follower_width > screen_width:
               new_window_x = screen_width - gaze_follower_width

          if new_window_y < 0:
               new_window_y = 0
          elif new_window_y + gaze_follower_height > screen_height:
               new_window_y = screen_height - gaze_follower_height

          cv2.namedWindow('Gaze Follower')
          cv2.resizeWindow('Gaze Follower', gaze_follower_width, gaze_follower_height)
          cv2.moveWindow('Gaze Follower', new_window_x, new_window_y)

# 눈의 좌표를 구하는 기능과 관련된 클래스
class EyeCoordinate:
     def __init__(self, tracker):
        self.tracker = tracker
        
     def find_centers(self, frame, face):
          """
          Find the centers of both eyes.

          Args:
               frame (numpy.ndarray): Image frame.
               face (dlib.rectangle): Detected face.

          Returns:
               tuple: A tuple containing the coordinates (x, y, width, height) of both eyes.
          """
          landmarks = self.tracker.predictor(frame, face)
          left_eye = landmarks.part(36).x, landmarks.part(37).y, landmarks.part(39).x - landmarks.part(36).x, landmarks.part(41).y - landmarks.part(37).y
          right_eye = landmarks.part(42).x, landmarks.part(43).y, landmarks.part(45).x - landmarks.part(42).x, landmarks.part(47).y - landmarks.part(43).y
          return left_eye, right_eye
     
     def get_centers(self, eyes):
          """
          Get the coordinates of the centers of the eyes.
          
          Args:
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.
               
          Returns:
               list: A list containing the coordinates of the centers of both eyes.
          """
          eye_centers = [] # 리스트 변수 선언
          for eye in eyes:
               x, y, w, h = eye
               eye_center_x = x + w // 2
               eye_center_y = y + h // 2
               eye_centers.append([eye_center_x, eye_center_y])
          
          return eye_centers
     
     def get_sight(self, frame, eyes):
          """
          Get the coordinate of the estimated eye sight based on the centers of the eyes.

          Args:
               eyes (tuple): A tuple containing the coordinates (x, y, width, height) of both eyes.

          Returns:
               tuple: A tuple containing the estimated coordinate of the eye sight (x, y).
          """
          left_pupil_x, left_pupil_y, _ = pupil_detector.detect_pupil_using_black_detection(frame, eyes[0])
          right_pupil_x, right_pupil_y, _ = pupil_detector.detect_pupil_using_black_detection(frame, eyes[1])
          
          eye_sight_x = (left_pupil_x + right_pupil_x) // 2
          eye_sight_y = (left_pupil_y + right_pupil_y) // 2
          
          return eye_sight_x, eye_sight_y

class PupilDetector:
     def __init__(self, eye_tracker):
          self.eye_tracker = eye_tracker

     def detect_pupil_using_black_detection(self, frame, eye):
          """
          Detect the pupil within the eye region using black detection method.

          Args:
               frame (numpy.ndarray): Image frame.
               eye (tuple): A tuple containing the coordinates (x, y, width, height) of the eye.

          Returns:
               tuple: A tuple containing the coordinates of the detected pupil (x, y, radius).
          """
          x, y, w, h = eye
          eye_region = frame[y:y+h, x:x+w]

          # Convert eye region to HSV color space
          eye_hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

          # Define lower and upper bounds for black color in HSV
          lower_black = np.array([0, 0, 0])
          upper_black = np.array([180, 255, 100])

          # Threshold the HSV image to get only black colors
          black_mask = cv2.inRange(eye_hsv, lower_black, upper_black)

          # Find contours
          contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          # If no contours found, return center of eye region
          if len(contours) == 0:
               return (x + w // 2, y + h // 2, 0)  # Return radius as 0 when no pupil is detected

          # Get the largest contour (presumably the pupil)
          largest_contour = max(contours, key=cv2.contourArea)

          # Get the moments of the largest contour
          M = cv2.moments(largest_contour)

          # Check if the contour area is 0
          if M["m00"] == 0:
               return (x + w // 2, y + h // 2, 0)  # Return radius as 0 when no pupil is detected

          # Calculate the centroid of the contour
          cx = int(M["m10"] / M["m00"])
          cy = int(M["m01"] / M["m00"])

          # Translate centroid coordinates to eye region coordinates
          pupil_x = x + cx
          pupil_y = y + cy

          # Calculate radius as half of eye region width
          radius = w // 5

          return (pupil_x, pupil_y, radius)



# 웹캠 시작
cap = cv2.VideoCapture(0)

# 클래스 초기화
eye_tracker = EyeTracker()
coordinate = EyeCoordinate(eye_tracker)
pupil_detector = PupilDetector(eye_tracker)

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
     faces = eye_tracker.detector(gray)

     for face in faces:
          eyes = coordinate.find_centers(frame, face) # 양쪽 눈의 중심 좌표 구하기
          eye_tracker.draw_face_landmarks(frame, face)
          eye_tracker.draw_eye_centers(frame, eyes) # 눈 좌표 그리기
          eye_tracker.draw_eye_sight(frame, eyes) # 시선 좌표 그리기
          eye_tracker.detect_and_draw_pupils(frame, eyes)  # 눈동자 감지 및 그리기
          eye_tracker.follow_gaze_window(frame, eyes, 75.0) # 시선을 추적하는 윈도우창을 띄우기
               
     # 결과 출력
     cv2.imshow('Eye Tracker', frame)

     # 'q'를 누르면 종료
     if cv2.waitKey(1) == 27:
          break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
