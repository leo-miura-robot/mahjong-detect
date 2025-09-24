import sys
from ultralytics import YOLO

model = YOLO("best2.pt")

image_path = sys.argv[1]

results = model.predict(source=image_path, conf=0.3, iou=0.3, imgsz=960)

order = [
    "1_man","2_man","3_man","4_man","5_man","6_man","7_man","8_man","9_man",
    "1_pin","2_pin","3_pin","4_pin","5_pin","6_pin","7_pin","8_pin","9_pin",
    "1_sou","2_sou","3_sou","4_sou","5_sou","6_sou","7_sou","8_sou","9_sou",
    "east","south","west","north","haku","hatsu","chun"
]

labels = []
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        labels.append(model.names[cls_id])

labels_sorted = sorted(labels, key=lambda x: order.index(x) if x in order else 999)

if labels_sorted:
    print("pai:", " ".join(labels_sorted))
else:
    print("ないよ")
