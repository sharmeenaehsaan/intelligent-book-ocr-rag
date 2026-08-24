import json
from pathlib import Path


def save_ocr_json(results, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    serializable_results = []

    for item in results:
        serializable_results.append({
            "text": item["text"],
            "confidence": float(item["confidence"]),
            "bbox": [
                [float(point[0]), float(point[1])]
                for point in item["bbox"]
            ],
            "left": float(item["left"]),
            "top": float(item["top"]),
            "right": float(item["right"]),
            "bottom": float(item["bottom"]),
            "width": float(item["width"]),
            "height": float(item["height"])
        })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            serializable_results,
            file,
            indent=4,
            ensure_ascii=False
        )