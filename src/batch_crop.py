from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Union

import cv2

from face_region_extractor import (
    FaceDetectionError,
    FaceNotFoundError,
    FaceRegionExtractor,
)

basicConfig(level=INFO)
logger = getLogger(__name__)


def batch_crop(
    input_dir: Union[str, Path], output_dir: Union[str, Path], glob="**/*.jpg"
):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not directory")

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError(f"{output_dir} is not directory")

    extractor = FaceRegionExtractor()

    for img_file in input_dir.glob(glob):
        logger.info(f"process {img_file}...")
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        try:
            face_imgs = extractor.extract(img)
        except FaceDetectionError as e:
            logger.warning(f"{e}. file: {img_file}")
            continue
        except FaceNotFoundError as e:
            logger.warning(f"{e}. file: {img_file}")
            continue

        n_faces = len(face_imgs)
        if n_faces > 1:
            logger.info(f"Two more faces are detected. skip: {img_file}")
            continue

        out_file = output_dir / img_file.relative_to(input_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), face_imgs[0])
        logger.info(f"save to {out_file}.")


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    batch_crop(args.input_dir, args.output_dir)
