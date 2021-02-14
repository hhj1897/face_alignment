import cv2
import numpy as np
from typing import Optional, Iterable


def plot_landmarks(image: np.ndarray, landmarks: np.ndarray, landmark_scores: Optional[Iterable[float]] = None,
                   threshold: float = 0.2, line_colour: Iterable[int] = (0, 255, 0),
                   pts_colour: Iterable[int] = (0, 0, 255), line_thickness: int = 1, pts_radius: int = 1) -> None:
    if landmarks.shape[0] > 0:
        if landmark_scores is None:
            landmark_scores = np.full(shape=(landmarks.shape[0],), fill_value=threshold + 1)
        if landmarks.shape[0] == 68:
            for idx in range(len(landmarks) - 1):
                if idx not in [16, 21, 26, 30, 35, 41, 47, 59]:
                    if landmark_scores[idx] >= threshold and landmark_scores[idx + 1] >= threshold:
                        cv2.line(image, tuple(landmarks[idx].astype(int).tolist()),
                                 tuple(landmarks[idx + 1].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
                if idx == 30:
                    if landmark_scores[30] >= threshold and landmark_scores[33] >= threshold:
                        cv2.line(image, tuple(landmarks[30].astype(int).tolist()),
                                 tuple(landmarks[33].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
                elif idx == 36:
                    if landmark_scores[36] >= threshold and landmark_scores[41] >= threshold:
                        cv2.line(image, tuple(landmarks[36].astype(int).tolist()),
                                 tuple(landmarks[41].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
                elif idx == 42:
                    if landmark_scores[42] >= threshold and landmark_scores[47] >= threshold:
                        cv2.line(image, tuple(landmarks[42].astype(int).tolist()),
                                 tuple(landmarks[47].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
                elif idx == 48:
                    if landmark_scores[48] >= threshold and landmark_scores[59] >= threshold:
                        cv2.line(image, tuple(landmarks[48].astype(int).tolist()),
                                 tuple(landmarks[59].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
                elif idx == 60:
                    if landmark_scores[60] >= threshold and landmark_scores[67] >= threshold:
                        cv2.line(image, tuple(landmarks[60].astype(int).tolist()),
                                 tuple(landmarks[67].astype(int).tolist()),
                                 color=line_colour, thickness=line_thickness, lineType=cv2.LINE_AA)
        for landmark, score in zip(landmarks, landmark_scores):
            if score >= threshold:
                cv2.circle(image, tuple(landmark.astype(int).tolist()), pts_radius, pts_colour, -1)
