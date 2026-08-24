import cv2
import numpy as np


def draw_ocr_boxes(image, ocr_results):
    if image.ndim == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()

    for index, item in enumerate(ocr_results, start=1):
        box = np.array(item["bbox"], dtype=np.int32)

        confidence = item["confidence"]

        if confidence >= 0.90:
            color = (0, 180, 0)
        elif confidence >= 0.70:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.polylines(
            output_image,
            [box],
            isClosed=True,
            color=color,
            thickness=2
        )

        label_x = int(item["left"])
        label_y = max(20, int(item["top"]) - 6)

        label = f"{index}: {confidence:.0%}"

        cv2.putText(
            output_image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return output_image