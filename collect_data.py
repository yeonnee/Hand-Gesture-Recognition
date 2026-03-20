import os
import cv2
import mediapipe as mp
import numpy as np
import random
import time


name = input("영문 이름: ")
folder_path = "./motion_data/" + name              
os.makedirs(folder_path, exist_ok=True)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils             # landmark 그려주는 함수 패키지
mp_drawing_styles = mp.solutions.drawing_styles     # 그리는 스타일 패키지
hands = mp_hands.Hands(
    max_num_hands=2,                                # 손 최대 갯수
    model_complexity=1,                             # 모델 복잡성
    min_detection_confidence=0.5,                   # 최소 탐지 신뢰도
    min_tracking_confidence=0.5)                    # 최소 추적 신뢰도

webcam = cv2.VideoCapture(0)

# 제스처 정의 
gesture={
    0 : 'Select Drone',
    1 : 'Select Group',
    2 : 'Select Mode',
    3 : 'ARM',
    4 : 'DISARM',
    5 : 'TAKEOFF',
    6 : 'LAND',
    7 : 'RTL',
    8 : 'Change Altitude',
    9 : 'Change Speed',
    10 : 'Move Up',
    11 : 'Move Down',
    12 : 'Move Forward',
    13 : 'Move Backward',
    14 : 'Move Left',
    15 : 'Move Right',
    16 : 'Rotate CW',
    17 : 'Rotate CCW',
    18 : 'Check',
    19 : 'Cancel'
}
gesture_keys = list(gesture.keys())
random.shuffle(gesture_keys)

for i in range(len(gesture)):
    key = gesture_keys.pop()
    print(gesture[key])
    input("Press Enter to start")

    folder_path2 = folder_path + "/" + gesture[key]
    os.makedirs(folder_path2, exist_ok=True)

    time.sleep(3)
    start_time = time.time()
    # print(start_time)

    idx = 1
    seq = []
    init_array = np.array(range(101), dtype='float64')

    while webcam.isOpened():
        status, frame = webcam.read()

        if not status:
            print("Not found")
            continue

        frame = cv2.flip(frame,1)                       # 보기 편하게 좌우반전
        frame.flags.writeable = False                   # frame을 읽기 전용으로 만들어줌
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)

        frame.flags.writeable = True                    # frame을 읽기/쓰기로 만들어줌
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for idx in range(len(result.multi_handedness)):
                joint = np.zeros((21,3))
                for j, lm in enumerate(result.multi_hand_landmarks[idx].landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]    # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]

                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 
                
                angle = np.degrees(angle) # Convert radian to degree

                angle_label = np.array([angle], dtype=np.float32)

                hand_label = result.multi_handedness[idx].classification[0].label
                angle_label = np.append(angle,hand_label)

                joint_angle_label = np.concatenate([joint.flatten(), angle_label])

                seq.append(joint_angle_label)

                mp_drawing.draw_landmarks(
                    frame, 
                    idx, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(), 
                    mp_drawing_styles.get_default_hand_connections_style())
    
            data = np.array(seq)
            # print(data)
            cv2.imwrite(folder_path2 + '/' + str(idx) + '.jpg', frame)
            idx += 1
            
            init_array = np.vstack((init_array, data))

        stamp_time = time.time() - start_time

        cv2.imshow('Data Collection for Gesture', frame)
        cv2.moveWindow('Data Collection for Gesture', 700,400)

        if cv2.waitKey(1) == ord('q'):
            break
        
        if stamp_time > 20:
            np.savetxt(folder_path2 + '/' + name + '_' + gesture[key] + '.csv', init_array[1:-20,:], delimiter=',', fmt='%s')
            break 

    cv2.destroyAllWindows()

webcam.release()