import os
import time
import argparse
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from face_detection.yolov5_face.detector import Yolov5Face
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference

def augment_image(img):
    """Generate simple augmentations that preserve identity."""
    h, w = img.shape[:2]
    aug = [img]
    # horizontal flip
    aug.append(cv2.flip(img, 1))
    # small rotations
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug.append(cv2.warpAffine(img, M, (w, h)))
    # brightness adjustments
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    for factor in (0.8, 1.2):
        v2 = np.clip(v_.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        img2 = cv2.cvtColor(cv2.merge([h_, s_, v2]), cv2.COLOR_HSV2BGR)
        aug.append(img2)
    return aug


def get_feature(face_img, recognizer, device):
    """Preprocess a face and return a normalized embedding."""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)
    emb = recognizer(tensor)[0].detach().cpu().numpy()
    return emb / np.linalg.norm(emb)


def build_templates(ref_dir, detector, recognizer, device):
    """Compute embeddings for augmented reference images and return list of templates."""
    templates = []
    for img_name in sorted(os.listdir(ref_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(ref_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for aug_img in augment_image(img):
            boxes, _ = detector.detect(image=aug_img)
            if boxes is None or len(boxes) == 0:
                continue
            x1, y1, x2, y2, _ = boxes[0]
            face = aug_img[int(y1):int(y2), int(x1):int(x2)]
            templates.append(get_feature(face, recognizer, device))
    if not templates:
        raise RuntimeError(f"No faces found for augmentation in {ref_dir}")
    return templates


def run_detection(args):
    # validate weight files
    if not os.path.isfile(args.detector_path):
        raise FileNotFoundError(
            f"Detector weights not found at {args.detector_path}."
        )
    if not os.path.isfile(args.recognizer_path):
        raise FileNotFoundError(
            f"ArcFace weights not found at {args.recognizer_path}."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = SCRFD(model_file=args.detector_path) if args.detector == 'scrfd' else Yolov5Face(model_file=args.detector_path)
    recognizer = iresnet_inference(model_name=args.model_name, path=args.recognizer_path, device=device)

    if not os.path.isdir(args.ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {args.ref_dir}")
    templates = build_templates(args.ref_dir, detector, recognizer, device)
    print(f"[INFO] Built {len(templates)} template embeddings from {args.ref_dir}")

    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    print("[INFO] Running augmented verification. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        boxes, _ = detector.detect(image=frame)
        num_faces = len(boxes) if boxes is not None else 0

        # Display face count on screen
        cv2.putText(frame, f"Faces: {num_faces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if boxes is not None and num_faces > 0:
            for (x1, y1, x2, y2, _) in boxes:
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                emb = get_feature(face, recognizer, device)
                sims = [cosine_similarity(emb.reshape(1,-1), tpl.reshape(1,-1))[0,0] for tpl in templates]
                sim = max(sims)
                if sim >= args.threshold:
                    color, label = (0,255,0), f"Target {sim:.2f}"
                else:
                    color, label = (0,0,255), f"Unknown {sim:.2f}"
                    print(f"Unknown face detected (max_sim={sim:.2f})")
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        writer.write(frame)
        cv2.imshow('Augmented Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augmented verification of a single target person.")
    parser.add_argument('--ref-dir', default='./datasets/target_person', help='Dir of reference images')
    parser.add_argument('--source', default='0', help='Camera index or video file')
    parser.add_argument('--detector', choices=['yolov5','scrfd'], default='yolov5', help='Detector to use')
    parser.add_argument('--detector-path', default='face_detection/yolov5_face/weights/yolov5m-face.pt', help='Path to detector weights')
    parser.add_argument('--recognizer-path', default='face_recognition/arcface/weights/arcface_r100.pth', help='Path to recognizer weights')
    parser.add_argument('--model-name', default='r100', help='ArcFace backbone')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    parser.add_argument('--output', default='results/output.mp4', help='Output video file')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_detection(args)