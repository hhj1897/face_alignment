import cv2


def plot_landmarks(frame, landmarks, line_colour=(0, 255, 0), pts_colour=(0, 0, 255),
                   line_thickness=1, pts_radius=1):
    if landmarks.shape[0] == 68:
        for idx in range(len(landmarks) - 1):
            if idx not in [16, 21, 26, 30, 35, 41, 47, 59]:
                cv2.line(frame, tuple(landmarks[idx].astype(int).tolist()),
                         tuple(landmarks[idx + 1].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
            if idx == 30:
                cv2.line(frame, tuple(landmarks[30].astype(int).tolist()),
                         tuple(landmarks[33].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
            elif idx == 36:
                cv2.line(frame, tuple(landmarks[36].astype(int).tolist()),
                         tuple(landmarks[41].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
            elif idx == 42:
                cv2.line(frame, tuple(landmarks[42].astype(int).tolist()),
                         tuple(landmarks[47].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
            elif idx == 48:
                cv2.line(frame, tuple(landmarks[48].astype(int).tolist()),
                         tuple(landmarks[59].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
            elif idx == 60:
                cv2.line(frame, tuple(landmarks[60].astype(int).tolist()),
                         tuple(landmarks[67].astype(int).tolist()),
                         color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
    for landmark in landmarks:
        cv2.circle(frame, tuple(landmark.astype(int).tolist()), pts_radius, pts_colour, -1)
