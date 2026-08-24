from pathlib import Path

import cv2


def point_inside_box(x, y, box):
    box_x, box_y, box_width, box_height = box

    return (
        box_x <= x <= box_x + box_width
        and box_y <= y <= box_y + box_height
    )


def block_center(block):
    center_x = (
        block["left"] + block["right"]
    ) / 2

    center_y = (
        block["top"] + block["bottom"]
    ) / 2

    return center_x, center_y


def get_blocks_inside_box(box, blocks):
    inside = []

    for block in blocks:
        center_x, center_y = block_center(block)

        if point_inside_box(
            center_x,
            center_y,
            box
        ):
            inside.append(block)

    return inside


def intersection_area(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    left = max(x1, x2)
    top = max(y1, y2)

    right = min(
        x1 + width1,
        x2 + width2
    )

    bottom = min(
        y1 + height1,
        y2 + height2
    )

    if right <= left or bottom <= top:
        return 0

    return (
        right - left
    ) * (
        bottom - top
    )


def boxes_overlap(box1, box2):
    return intersection_area(
        box1,
        box2
    ) > 0


def merge_two_boxes(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    left = min(x1, x2)
    top = min(y1, y2)

    right = max(
        x1 + width1,
        x2 + width2
    )

    bottom = max(
        y1 + height1,
        y2 + height2
    )

    return (
        left,
        top,
        right - left,
        bottom - top
    )


def merge_overlapping_boxes(boxes):
    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        output = []

        while boxes:
            current = boxes.pop(0)

            index = 0

            while index < len(boxes):
                if boxes_overlap(
                    current,
                    boxes[index]
                ):
                    current = merge_two_boxes(
                        current,
                        boxes[index]
                    )

                    boxes.pop(index)
                    changed = True
                else:
                    index += 1

            output.append(current)

        boxes = output

    return boxes


def detect_figure_regions(
    image,
    classified_blocks,
    debug=False
):
    if image is None:
        raise ValueError(
            "Figure detector received an empty image."
        )

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    page_height, page_width = gray.shape[:2]
    page_area = page_width * page_height

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    connected = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, width, height = (
            cv2.boundingRect(contour)
        )

        if width <= 0 or height <= 0:
            continue

        area_ratio = (
            width * height
        ) / page_area

        aspect_ratio = (
            width / height
        )

        if debug:
            print(
                f"x={x}, y={y}, "
                f"w={width}, h={height}, "
                f"area={area_ratio:.4f}, "
                f"aspect={aspect_ratio:.2f}"
            )

        if area_ratio < 0.004:
            continue

        if area_ratio > 0.20:
            continue

        if width < page_width * 0.12:
            continue

        if height < page_height * 0.035:
            continue

        if aspect_ratio < 0.55:
            continue

        if aspect_ratio > 4.0:
            continue

        if (
            x <= 3
            or y <= 3
            or x + width >= page_width - 3
            or y + height >= page_height - 3
        ):
            continue

        candidate = (
            x,
            y,
            width,
            height
        )

        inside_blocks = get_blocks_inside_box(
            candidate,
            classified_blocks
        )

        # A real screenshot, diagram, or chart commonly
        # contains multiple separately detected text pieces.
        if len(inside_blocks) < 2:
            continue

        # Reject a region that is mostly OCR text rather than
        # an actual photo/diagram. A real figure or screenshot
        # is sparse (maybe a caption); a paragraph or short
        # code sample is almost entirely covered by its own
        # text bounding boxes. Unlike a line-count-based check,
        # this works even for 2-3 line code blocks or short
        # lists, which used to slip through undetected.
        candidate_area = width * height

        text_area = sum(
            (block["right"] - block["left"])
            * (block["bottom"] - block["top"])
            for block in inside_blocks
        )

        text_coverage = (
            text_area / candidate_area
            if candidate_area > 0
            else 0
        )

        if text_coverage > 0.35:
            continue

        candidates.append(candidate)

    candidates = merge_overlapping_boxes(
        candidates
    )

    final_boxes = []

    for box in candidates:
        x, y, width, height = box

        area_ratio = (
            width * height
        ) / page_area

        if area_ratio > 0.20:
            continue

        final_boxes.append(box)

    final_boxes.sort(
        key=lambda box: (
            box[1],
            box[0]
        )
    )

    return final_boxes


def mark_blocks_inside_figures(
    classified_blocks,
    figure_boxes
):
    updated_blocks = []

    for block in classified_blocks:
        block_left = block["left"]
        block_top = block["top"]
        block_right = block["right"]
        block_bottom = block["bottom"]

        block_width = max(
            block_right - block_left,
            1
        )

        block_height = max(
            block_bottom - block_top,
            1
        )

        block_area = (
            block_width * block_height
        )

        inside_figure = False

        for figure_box in figure_boxes:
            x, y, width, height = figure_box

            # Shrink the figure box slightly so captions located
            # immediately outside the picture are not removed.
            margin_x = width * 0.03
            margin_y = height * 0.03

            figure_left = x + margin_x
            figure_top = y + margin_y
            figure_right = x + width - margin_x
            figure_bottom = y + height - margin_y

            intersection_left = max(
                block_left,
                figure_left
            )

            intersection_top = max(
                block_top,
                figure_top
            )

            intersection_right = min(
                block_right,
                figure_right
            )

            intersection_bottom = min(
                block_bottom,
                figure_bottom
            )

            intersection_width = max(
                0,
                intersection_right
                - intersection_left
            )

            intersection_height = max(
                0,
                intersection_bottom
                - intersection_top
            )

            intersection_area = (
                intersection_width
                * intersection_height
            )

            overlap_ratio = (
                intersection_area
                / block_area
            )

            block_center_x = (
                block_left + block_right
            ) / 2

            block_center_y = (
                block_top + block_bottom
            ) / 2

            center_inside = (
                figure_left
                <= block_center_x
                <= figure_right
                and figure_top
                <= block_center_y
                <= figure_bottom
            )

            if (
                center_inside
                and overlap_ratio >= 0.55
            ):
                inside_figure = True
                break

        updated = block.copy()

        if inside_figure:
            updated["type"] = "inside_figure"

        updated_blocks.append(updated)

    return updated_blocks


def save_figure_crops(
    image,
    figure_boxes,
    output_directory,
    padding=8
):
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_figures = []

    for index, box in enumerate(
        figure_boxes,
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

        if crop.size == 0:
            continue

        figure_path = (
            output_directory
            / f"figure_{index}.png"
        )

        if not cv2.imwrite(
            str(figure_path),
            crop
        ):
            raise IOError(
                f"Could not save {figure_path}"
            )

        saved_figures.append({
            "path": str(figure_path),
            "bbox": (
                left,
                top,
                right - left,
                bottom - top
            )
        })

    return saved_figures


def draw_figure_boxes(
    image,
    figure_boxes
):
    output = image.copy()

    for index, box in enumerate(
        figure_boxes,
        start=1
    ):
        x, y, width, height = box

        cv2.rectangle(
            output,
            (x, y),
            (
                x + width,
                y + height
            ),
            (255, 0, 255),
            3
        )

        cv2.putText(
            output,
            f"Figure {index}",
            (
                x,
                max(25, y - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    return output