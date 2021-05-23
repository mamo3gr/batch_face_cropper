import cv2

from face_region_extractor import FaceRegionExtractor


class TestFaceRegionExtractor:
    def test_extract(self):
        expand_margin = 16
        extractor = FaceRegionExtractor(expand_margin=expand_margin)
        img = cv2.imread("tests/resources/image.jpg", cv2.IMREAD_COLOR)

        face_imgs = extractor.extract(img)
        assert len(face_imgs) > 0
