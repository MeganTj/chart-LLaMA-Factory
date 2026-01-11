# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time

import fire
from datasets import load_dataset
from torchvision.ops.boxes import box_area
import torch
import re


def calc_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union

def parse_bbox_from_text(text):
    """
    Parse bounding box coordinates from various text formats.
    Returns list of coordinates [x1, y1, x2, y2] or None if not found.
    Finds the first sequence of exactly 4 numbers (integers or floats) surrounded by brackets or parentheses.
    """
    # Find all bracket/parenthesis pairs and their content
    patterns = [
        r'\[([^\]]+)\]',  # Content within square brackets
        r'\(([^)]+)\)'    # Content within parentheses
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            content = match.group(1)
            # Extract all numbers (integers or floats) from the content
            numbers = re.findall(r'\d+\.?\d*', content)
            
            if len(numbers) == 4:  # Check for exactly 4 numbers
                return [float(x) for x in numbers]
    
    return None

def compute_metrics(sample):
    # Convert prediction to bounding box
    gt_bbox = json.loads(sample["label"])
    predicted_bbox = parse_bbox_from_text(sample["predict"])
    if predicted_bbox is not None:
        iou = calc_iou(gt_bbox, predicted_bbox)
        return {"iou": iou, "acc": float(iou > 0.5)}
    else:
        print("No bounding box found")
        return {"iou": 0.0, "acc": 0.0}


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    dataset = dataset.map(compute_metrics, num_proc=8, remove_columns=dataset.column_names)
    score_dict = dataset.to_dict()

    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)

    with open("predictions_score.json", "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to predictions_score.json")


if __name__ == "__main__":
    fire.Fire(main)
