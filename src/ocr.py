import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCREngine:

    def __init__(self, language: str = "en"):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=language,
            show_log=False,
            det_db_thresh=0.20,
            det_db_box_thresh=0.30,
            det_db_unclip_ratio=1.8
        )

    def prepare_image(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> np.ndarray:

        if isinstance(image, Path):
            image = str(image)

        if isinstance(image, str):
            loaded_image = cv2.imread(image)

            if loaded_image is None:
                raise FileNotFoundError(
                    f"Could not load image: {image}"
                )

            image = loaded_image

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "OCR input must be an image path "
                "or a NumPy array."
            )

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        if image.ndim != 3:
            raise ValueError(
                "OCR image must be grayscale "
                "or a three-channel image."
            )

        return image

    def extract_once(
        self,
        image: np.ndarray,
        y_offset: int = 0
    ) -> list[dict]:

        results = self.ocr.ocr(
            image,
            cls=True
        )

        output = []

        if not results:
            return output

        for page in results:
            if page is None:
                continue

            for line in page:
                if not line or len(line) < 2:
                    continue

                box = line[0]
                recognition = line[1]

                if not recognition or len(recognition) < 2:
                    continue

                text = str(
                    recognition[0]
                ).strip()

                confidence = float(
                    recognition[1]
                )

                if not text:
                    continue

                adjusted_box = [
                    [
                        float(point[0]),
                        float(point[1] + y_offset)
                    ]
                    for point in box
                ]

                x_values = [
                    point[0]
                    for point in adjusted_box
                ]

                y_values = [
                    point[1]
                    for point in adjusted_box
                ]

                left = min(x_values)
                top = min(y_values)
                right = max(x_values)
                bottom = max(y_values)

                output.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": adjusted_box,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top
                })

        return output

    def normalized_text(
        self,
        text: str
    ) -> str:

        return "".join(
            character.lower()
            for character in text
            if character.isalnum()
        )

    def horizontal_overlap(
        self,
        first: dict,
        second: dict
    ) -> float:

        overlap_left = max(
            first["left"],
            second["left"]
        )

        overlap_right = min(
            first["right"],
            second["right"]
        )

        overlap_width = max(
            0,
            overlap_right - overlap_left
        )

        smaller_width = max(
            1,
            min(
                first["width"],
                second["width"]
            )
        )

        return overlap_width / smaller_width

    def vertical_center_difference(
        self,
        first: dict,
        second: dict
    ) -> float:

        first_center = (
            first["top"]
            + first["bottom"]
        ) / 2

        second_center = (
            second["top"]
            + second["bottom"]
        ) / 2

        return abs(
            first_center - second_center
        )

    def are_duplicates(
        self,
        first: dict,
        second: dict
    ) -> bool:

        center_difference = (
            self.vertical_center_difference(
                first,
                second
            )
        )

        allowed_difference = max(
            12,
            min(
                first["height"],
                second["height"]
            ) * 0.70
        )

        if center_difference > allowed_difference:
            return False

        overlap_ratio = self.horizontal_overlap(
            first,
            second
        )

        if overlap_ratio < 0.30:
            return False

        first_text = self.normalized_text(
            first["text"]
        )

        second_text = self.normalized_text(
            second["text"]
        )

        if not first_text or not second_text:
            return False

        shorter_text = min(
            first_text,
            second_text,
            key=len
        )

        longer_text = max(
            first_text,
            second_text,
            key=len
        )

        contained = (
            len(shorter_text) >= 7
            and shorter_text in longer_text
        )

        similarity = SequenceMatcher(
            None,
            first_text,
            second_text
        ).ratio()

        return (
            contained
            or similarity >= 0.68
        )

    def result_quality_score(
        self,
        item: dict
    ) -> float:

        text_length = max(
            len(item["text"].strip()),
            1
        )

        return (
            item["confidence"]
            * text_length
        )

    def merge_duplicate_results(
        self,
        results: list[dict]
    ) -> list[dict]:

        sorted_results = sorted(
            results,
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        merged_results = []

        for current in sorted_results:
            duplicate_index = None

            for index, existing in enumerate(
                merged_results
            ):
                if self.are_duplicates(
                    current,
                    existing
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                merged_results.append(
                    current
                )

                continue

            existing = merged_results[
                duplicate_index
            ]

            current_score = (
                self.result_quality_score(
                    current
                )
            )

            existing_score = (
                self.result_quality_score(
                    existing
                )
            )

            if current_score > existing_score:
                merged_results[
                    duplicate_index
                ] = current

        merged_results.sort(
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        return merged_results

    def remove_spatial_duplicates(
        self,
        results: list[dict]
    ) -> list[dict]:

        sorted_results = sorted(
            results,
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        kept_results = []

        for current in sorted_results:
            duplicate_index = None

            for index, existing in enumerate(
                kept_results
            ):
                center_difference = (
                    self.vertical_center_difference(
                        current,
                        existing
                    )
                )

                allowed_difference = max(
                    10,
                    min(
                        current["height"],
                        existing["height"]
                    ) * 0.60
                )

                if center_difference > allowed_difference:
                    continue

                overlap_ratio = (
                    self.horizontal_overlap(
                        current,
                        existing
                    )
                )

                if overlap_ratio < 0.20:
                    continue

                current_text = (
                    self.normalized_text(
                        current["text"]
                    )
                )

                existing_text = (
                    self.normalized_text(
                        existing["text"]
                    )
                )

                if not current_text or not existing_text:
                    continue

                shorter_text = min(
                    current_text,
                    existing_text,
                    key=len
                )

                longer_text = max(
                    current_text,
                    existing_text,
                    key=len
                )

                length_ratio = (
                    len(shorter_text)
                    / max(len(longer_text), 1)
                )

                similarity = SequenceMatcher(
                    None,
                    current_text,
                    existing_text
                ).ratio()

                is_fragment = (
                    len(shorter_text) >= 3
                    and (
                        shorter_text in longer_text
                        or length_ratio < 0.65
                    )
                )

                if (
                    is_fragment
                    or similarity >= 0.60
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                kept_results.append(
                    current
                )

                continue

            existing = kept_results[
                duplicate_index
            ]

            current_score = (
                self.result_quality_score(
                    current
                )
            )

            existing_score = (
                self.result_quality_score(
                    existing
                )
            )

            if current_score > existing_score:
                kept_results[
                    duplicate_index
                ] = current

        kept_results.sort(
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        return kept_results

    def is_page_number(
        self,
        item: dict,
        image_height: int
    ) -> bool:

        text = item["text"].strip()

        near_bottom = (
            item["bottom"]
            > image_height * 0.86
        )

        if not near_bottom:
            return False

        numeric_page_number = bool(
            re.fullmatch(
                r"\d{1,4}",
                text
            )
        )

        roman_page_number = bool(
            re.fullmatch(
                r"[ivxlcdm]+",
                text,
                flags=re.IGNORECASE
            )
        )

        return (
            numeric_page_number
            or roman_page_number
        )

    def is_suspicious_result(
        self,
        item: dict,
        image_height: int
    ) -> bool:

        text = item["text"].strip()
        confidence = item["confidence"]

        if not text:
            return True

        if self.is_page_number(
            item,
            image_height
        ):
            return False

        words = text.split()

        alphabetic_count = sum(
            character.isalpha()
            for character in text
        )

        alphanumeric_count = sum(
            character.isalnum()
            for character in text
        )

        alpha_ratio = (
            alphabetic_count
            / max(len(text), 1)
        )

        alphanumeric_ratio = (
            alphanumeric_count
            / max(len(text), 1)
        )

        near_top = (
            item["top"]
            < image_height * 0.18
        )

        possible_title = (
            near_top
            and len(text) >= 4
            and confidence >= 0.75
        )

        if possible_title:
            return False

        if confidence < 0.60:
            return True

        # Examples:
        # D, DOLN, orex
        if (
            confidence < 0.82
            and len(words) <= 2
            and len(text) <= 12
        ):
            return True

        # Examples:
        # Ilysdwayllonn, OSNIASSOOA
        if (
            confidence < 0.86
            and len(words) == 1
            and len(text) >= 8
        ):
            return True

        # Reject short symbol-heavy fragments.
        if (
            confidence < 0.85
            and alphanumeric_ratio < 0.65
        ):
            return True

        # Reject lines containing too little readable text.
        if (
            confidence < 0.80
            and alpha_ratio < 0.50
        ):
            return True

        return False

    def extract(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> list[dict]:

        image = self.prepare_image(
            image
        )

        image_height = image.shape[0]

        all_results = []

        full_page_results = self.extract_once(
            image,
            y_offset=0
        )

        all_results.extend(
            full_page_results
        )

        strip_height = max(
            500,
            int(image_height * 0.35)
        )

        overlap = int(
            strip_height * 0.25
        )

        step = max(
            200,
            strip_height - overlap
        )

        strip_number = 1
        strip_top = 0

        while strip_top < image_height:
            strip_bottom = min(
                image_height,
                strip_top + strip_height
            )

            crop = image[
                strip_top:strip_bottom,
                :
            ]

            if crop.shape[0] < 100:
                break

            strip_results = self.extract_once(
                crop,
                y_offset=strip_top
            )

            print(
                f"OCR strip {strip_number}: "
                f"top={strip_top}, "
                f"bottom={strip_bottom}, "
                f"regions={len(strip_results)}"
            )

            all_results.extend(
                strip_results
            )

            if strip_bottom >= image_height:
                break

            strip_top += step
            strip_number += 1

        merged_results = (
            self.merge_duplicate_results(
                all_results
            )
        )

        spatially_cleaned_results = (
            self.remove_spatial_duplicates(
                merged_results
            )
        )

        filtered_results = [
            item
            for item in spatially_cleaned_results
            if not self.is_suspicious_result(
                item,
                image_height
            )
        ]

        filtered_results.sort(
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        print(
            f"Full-page OCR regions: "
            f"{len(full_page_results)}"
        )

        print(
            f"Merged OCR regions: "
            f"{len(merged_results)}"
        )

        print(
            f"After spatial cleanup: "
            f"{len(spatially_cleaned_results)}"
        )

        print(
            f"After noise filtering: "
            f"{len(filtered_results)}"
        )

        return filtered_results