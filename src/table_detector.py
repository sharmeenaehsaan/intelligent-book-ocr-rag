from pathlib import Path

import cv2
import numpy as np


def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return not (
        x1 + w1 < x2
        or x2 + w2 < x1
        or y1 + h1 < y2
        or y2 + h2 < y1
    )


def merge_boxes(boxes):
    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        merged = []

        while boxes:
            current = boxes.pop(0)

            index = 0

            while index < len(boxes):
                other = boxes[index]

                if boxes_overlap(current, other):
                    x1 = min(current[0], other[0])
                    y1 = min(current[1], other[1])

                    x2 = max(
                        current[0] + current[2],
                        other[0] + other[2]
                    )

                    y2 = max(
                        current[1] + current[3],
                        other[1] + other[3]
                    )

                    current = (
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    )

                    boxes.pop(index)
                    changed = True
                else:
                    index += 1

            merged.append(current)

        boxes = merged

    return boxes


def detect_table_regions(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    height, width = gray.shape

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, width // 20), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(30, height // 30))
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    # A real table has a GRID: horizontal AND vertical lines
    # that intersect. Plain text (underlines, justified rows,
    # dense paragraphs) can trigger horizontal OR vertical
    # detection on its own, so requiring both drastically
    # cuts false positives on ordinary book pages.
    intersections = cv2.bitwise_and(
        horizontal_lines,
        vertical_lines
    )

    intersection_pixels = cv2.countNonZero(
        intersections
    )

    # No real grid intersections at all -> not a table.
    if intersection_pixels < 40:
        return []

    table_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines
    )

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    table_mask = cv2.morphologyEx(
        table_mask,
        cv2.MORPH_CLOSE,
        connect_kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        table_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    page_area = width * height
    table_boxes = []

    for contour in contours:
        x, y, box_width, box_height = (
            cv2.boundingRect(contour)
        )

        area_ratio = (
            box_width * box_height
        ) / page_area

        if area_ratio < 0.03:
            continue

        if area_ratio > 0.80:
            continue

        if box_width < width * 0.35:
            continue

        # Raised from 0.06: a couple of lines of ordinary
        # text can be 6% of the page tall, but a real table
        # needs multiple rows to be worth rendering as one.
        if box_height < height * 0.10:
            continue

        # Confirm this specific contour region actually
        # contains grid intersections of its own, not just
        # that the page somewhere has a stray one.
        region_intersections = intersections[
            y : y + box_height,
            x : x + box_width
        ]

        if cv2.countNonZero(region_intersections) < 20:
            continue

        table_boxes.append(
            (x, y, box_width, box_height)
        )

    return merge_boxes(table_boxes)


def point_inside_box(x, y, box):
    box_x, box_y, box_width, box_height = box

    return (
        box_x <= x <= box_x + box_width
        and box_y <= y <= box_y + box_height
    )


def mark_blocks_inside_tables(
    classified_blocks,
    table_boxes
):
    updated_blocks = []

    for block in classified_blocks:
        center_x = (
            block["left"] + block["right"]
        ) / 2

        center_y = (
            block["top"] + block["bottom"]
        ) / 2

        inside_table = any(
            point_inside_box(
                center_x,
                center_y,
                table_box
            )
            for table_box in table_boxes
        )

        updated = block.copy()

        # Headings are short, deliberate, and rarely ever
        # part of a real table cell. Never let a (possibly
        # false-positive) table box delete a heading.
        if inside_table and block["type"] != "heading":
            updated["type"] = "inside_table"

        updated_blocks.append(updated)

    return updated_blocks


def save_table_crops(
    image,
    table_boxes,
    output_directory,
    padding=8
):
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_tables = []

    for index, box in enumerate(
        table_boxes,
        start=1
    ):
        x, y, width, height = box

        left = max(0, x - padding)
        top = max(0, y - padding)

        right = min(
            image.shape[1],
            x + width + padding
        )

        bottom = min(
            image.shape[0],
            y + height + padding
        )

        crop = image[
            top:bottom,
            left:right
        ]

        path = (
            output_directory
            / f"table_{index}.png"
        )

        if not cv2.imwrite(str(path), crop):
            raise IOError(
                f"Could not save table: {path}"
            )

        saved_tables.append({
            "path": str(path),
            "bbox": (
                left,
                top,
                right - left,
                bottom - top
            )
        })

    return saved_tables


def draw_table_boxes(image, table_boxes):
    output = image.copy()

    for index, box in enumerate(
        table_boxes,
        start=1
    ):
        x, y, width, height = box

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            3
        )

        cv2.putText(
            output,
            f"Table {index}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return output