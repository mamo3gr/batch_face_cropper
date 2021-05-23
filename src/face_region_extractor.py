from logging import getLogger
from typing import List

import cv2
import numpy as np
from mtcnn import MTCNN
from mtcnn.exceptions.invalid_image import InvalidImage

from utils import set_gpu_memory_growth

set_gpu_memory_growth()

ARCFACE_LANDMARK = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceDetectionError(Exception):
    """Error while detecting faces in the image"""


class FaceNotFoundError(Exception):
    """No face is found in the input image"""


class FaceRegionExtractor:
    def __init__(self, logger=None, expand_margin=0):
        self._logger = logger or getLogger(__name__)
        self._detector = MTCNN()
        self._image_size = 112 + expand_margin * 2  # expand both side
        self._reference_landmarks = ARCFACE_LANDMARK + expand_margin

    def extract(self, img: np.ndarray) -> List[np.ndarray]:
        try:
            faces = self._detector.detect_faces(img)
        except InvalidImage as e:
            raise FaceDetectionError("error while detecting faces") from e

        if not faces:
            raise FaceNotFoundError("face not found")

        landmarks_for_each_face = [self.get_landmarks(face) for face in faces]
        face_imgs = [
            self.crop_face_image(img, landmarks)
            for landmarks in landmarks_for_each_face
        ]
        return face_imgs

    def crop_face_image(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self._reference_landmarks,
            method=cv2.LMEDS,
            confidence=0.999999,
            refineIters=100,
        )
        img_face = cv2.warpAffine(
            img, matrix, (self._image_size, self._image_size), borderValue=0
        )
        return img_face

    @staticmethod
    def get_landmarks(face: dict) -> np.ndarray:
        keypoints: dict = face["keypoints"]
        landmarks = np.array(
            [
                keypoints["left_eye"],
                keypoints["right_eye"],
                keypoints["nose"],
                keypoints["mouth_left"],
                keypoints["mouth_right"],
            ]
        )

        return landmarks
