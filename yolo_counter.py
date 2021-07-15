import argparse
import csv
from typing import Union, List, Optional

import cv2
import norfair
import numpy as np
import torch
import yolov5
from norfair import Detection, Tracker, Video
from yolov5.utils.plots import Colors

max_distance_between_points: int = 30
colors = Colors()  # create instance for 'from utils.plots import colors'

def convert_id_to_class(id_number: int):
    color_dict = {
        0 : "person",
        1 : "bicycle",
        2 : "car",
        80 : "cargovelo"
    }
    return color_dict.get(id_number)
class New_tracked_object:
    def __init__(self, identity, i):
        self.identity = identity
        self.id = self.convert_id_to_class(id)


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)
        self.names = self.model.names

    def __call__(
            self,
            img: Union[str, np.ndarray],
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            image_size: int = 720,
            classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor,
        track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores)
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores, data=int(detection_as_xyxy[5].item()))
            )
    # detections_as_pred = yolo_detections.pred[0]
    # for detection_as_pred in detections_as_pred:
    #     # for *xyxy, conf, cls in reversed(detection_as_pred):
    #     print(int(detection_as_pred[-1]))
    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("--files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)
detected_objects = {}
detected_classes = set()


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.conf_thres,
            image_size=args.img_size,
            classes=args.classes
        )
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
        tracked_objects = tracker.update(detections=detections)
        for d in detections:
            detected_classes.add(d.data)
        if args.track_points == 'centroid':
            norfair.draw_points(frame, detections)
        elif args.track_points == 'bbox':
            norfair.draw_boxes(frame, detections, line_width=3)
        if len(tracked_objects) > 0:
            for tracked_object in tracked_objects:
                detected_objects[tracked_object.id] = tracked_object
        new_tracked_objects=[]
        for d in detections:
            points = [d.points[0][0], d.points[0][1], d.points[1][0], d.points[1][1]]
            plot_one_box(points, frame, label=convert_id_to_class(d.data), color=colors(d.data, True), line_thickness=3)

    # for to in tracked_objects:
    #         new_tracked_objects.append(New_tracked_object(identity=to.id, id=to.last_detection.data))
        # norfair.draw_tracked_objects(frame, new_tracked_objects)
        video.write(frame)

counters = {0: 0, 1: 0, 2: 0}
size = len(str(args.files))
csv_file = open(str(args.files)[2:size-6]+'.csv', 'w')
writer = csv.writer(csv_file)
writer.writerow(["Object ID", "Class", "Object age on camera"])

for key in detected_objects:
    detected_object = detected_objects[key]
    writer.writerow([detected_object.id, detected_object.last_detection.data, detected_object.age])
    counters[detected_object.last_detection.data] = counters[detected_object.last_detection.data] + 1

summary_file = open(str(args.files)[2:size-6]+'_summary.txt', 'w')
summary_writer = csv.writer(summary_file)
for cls in detected_classes:
    print("detected: " + str(counters[cls]) + " " + model.names[cls])
    summary_writer.writerow([str(model.names[cls]), str(counters[cls])])

summary_file.close()
csv_file.close()