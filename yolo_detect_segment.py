import os, glob, json
import cv2
from ultralytics import YOLO

os.makedirs("out_det", exist_ok=True)
os.makedirs("out_seg", exist_ok=True)

model = YOLO("yolov8n-seg.pt")

frames = sorted(glob.glob("frames/*.png"))
det_results = {}

for fp in frames:
    name = os.path.basename(fp)
    img = cv2.imread(fp)
    r = model(img, verbose=False)[0]

    boxes = []
    if r.boxes is not None:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            boxes.append({
                "box": [float(v) for v in box],
                "label": model.names[int(cls)],
                "score": float(conf),
            })
    det_results[name] = boxes

    annotated = r.plot()
    cv2.imwrite(os.path.join("out_det", name), annotated)
    cv2.imwrite(os.path.join("out_seg", name), annotated)

    print(f"{name}: {len(boxes)} objects")

json.dump(det_results, open("detections.json", "w"), indent=2)
print("Done")


