import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from datetime import datetime
import time
import signal
import sys

import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

mongo_client = MongoClient("your Mongo DB String")
mongo_db = mongo_client["face_db"]
mongo_summary_col = mongo_db["face_embeddings_for_rtsp_testing"]
embedding_collection = mongo_db["face_vectors_raw"]  # NEW: Attendance collection
log_collection = mongo_db["face_access_logs"]
attendance_collection = mongo_db["attendance_logs"]

# === Load Models ===
def load_yolo_model():
    return YOLO("path to your engine/pt file ")

def load_arcface_model():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# === Image Enhancements ===
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.1, beta=3)

def is_front_facing(pose, yaw_threshold=25, pitch_threshold=25, roll_threshold=25):
    pitch, yaw, roll = pose
    return all(abs(angle) <= threshold for angle, threshold in zip((yaw, pitch, roll), (yaw_threshold, pitch_threshold, roll_threshold)))

# === Fetch Embeddings from MongoDB ===
def fetch_embedding_database_from_mongodb():
    db_embeds = {}
    for doc in embedding_collection.find():
        name = doc["name"]
        emb = np.array(doc["embedding"], dtype=np.float32)
        if name not in db_embeds:
            db_embeds[name] = []
        db_embeds[name].append(emb)
    print(f"Loaded embeddings from MongoDB: {len(db_embeds)} people")
    return db_embeds

# === Cosine Similarity Matching ===
def match_face(embedding, db_embeds, threshold=0.45):
    best_sim = 0
    best_name = "Unknown"
    for name, emb_list in db_embeds.items():
        for db_emb in emb_list:
            sim = cosine_similarity(embedding.reshape(1, -1), db_emb.reshape(1, -1))[0][0]
            if sim > best_sim:
                best_sim = sim
                best_name = name
    return (best_name, best_sim) if best_sim >= threshold else ("Unknown", best_sim)

# === Check if person already marked present today ===
def is_attendance_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    existing = attendance_collection.find_one({"name": name, "date": today})
    return existing is not None

# === Mark Attendance (First time only per day) ===
def mark_attendance(name, sim):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    attendance_collection.insert_one({
        "name": name,
        "date": today,
        "arrival_time": current_time,
        "timestamp": datetime.now(),
        "similarity": float(sim),
        "status": "Present"
    })
    print(f"✅ ATTENDANCE MARKED for {name} at {current_time}")

# === Insert Access Log (Keep for security tracking) ===
def log_access(name, sim, status):
    log_collection.insert_one({
        "name": name,
        "similarity": float(sim),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "access": status
    })

# === Live Pipeline ===
def process_video_live(input_stream, yolo_model, arcface_app, db_embeds, threshold=0.45):
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        print(" Failed to open RTSP stream.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"arcface_yolo_output_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

    frame_idx = 0
    trackers, identities = [], []
    marked_today = set()  # In-memory cache to avoid repeated DB queries

    print(" Press 'q' or Ctrl+C to stop...")

    try:
        while True:
            total_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or camera disconnected.")
                break

            enhanced = enhance_image(frame)

            if frame_idx % 5 == 0:
                detection_start = time.perf_counter()
                #YOLO detects faces
                results = yolo_model(enhanced)
                detection_time = (time.perf_counter() - detection_start) * 1000

                #Extract bounding boxes
                boxes = results[0].boxes
                #ArcFace embedding extraction only for those boxes:
                faces = arcface_app.get(enhanced)

                trackers.clear()
                identities.clear()
                recognition_time = 0  # default

                if boxes and faces:
                    #Match YOLO detections to ArcFace faces
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if conf < 0.5 or cls != 0:
                            continue

                        best_face, best_iou = None, 0
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            inter_area = max(0, min(x2, fx2) - max(x1, fx1)) * max(0, min(y2, fy2) - max(y1, fy1))
                            union_area = (x2 - x1) * (y2 - y1) + (fx2 - fx1) * (fy2 - fy1) - inter_area
                            iou = inter_area / union_area if union_area > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_face = face

                        if best_face is None or best_iou < 0.2 or not is_front_facing(best_face.pose):
                            continue

                        recognition_start = time.perf_counter()
                        #Generate Embedding and Compare with Database
                        emb = best_face.embedding
                        emb /= np.linalg.norm(emb)
                        name, sim = match_face(emb, db_embeds, threshold)
                        recognition_time = (time.perf_counter() - recognition_start) * 1000
                        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if name != "Unknown":
                            # Check if attendance already marked today
                            if name not in marked_today:
                                if not is_attendance_marked_today(name):
                                    mark_attendance(name, sim)
                                    marked_today.add(name)
                                else:
                                    marked_today.add(name)
                                    print(f"Already Present: {name} ({sim:.2f})")
                            
                            # Still log access for security purposes
                            log_access(name, sim, "GRANTED")
                        else:
                            log_access("Unknown", sim, "DENIED")
                            print(f"ACCESS DENIED - Unknown person detected at {time_stamp}")
                        
                        #tracker = cv2.TrackerCSRT_create()
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                        trackers.append(tracker)
                        identities.append((name, sim))

            else:
                new_boxes = []
                for idx, tracker in enumerate(trackers):
                    ok, bbox = tracker.update(frame)
                    if ok:
                        x, y, w, h = map(int, bbox)
                        new_boxes.append((x, y, x + w, y + h, identities[idx]))

                for x1, y1, x2, y2, (name, sim) in new_boxes:
                    color = (0, 255, 0) if name != "Unknown" else (0, 255, 0)
                    label = f"{name} ({sim:.2f})" if name != "Unknown" else "Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            resized_frame = cv2.resize(frame, (1024, 720))
            cv2.imshow("Live CCTV Face Recognition", resized_frame)
            out.write(cv2.resize(frame, (640, 640)))

            total_time = (time.perf_counter() - total_start) * 1000
            if frame_idx % 5 == 0:
                print(f" Detection: {detection_time:.2f} ms | Recognition: {recognition_time:.2f} ms | Total: {total_time:.2f} ms")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" Manual stop by user.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        print("Interrupted by Ctrl+C")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f" CCTV closed and video saved to: {output_path}")

# === Entry Point ===
if __name__ == "__main__":
    # === Choose between webcam or RTSP stream ===
    USE_WEBCAM = True  # Set to False to use RTSP

    if USE_WEBCAM:
        input_stream = 0  # Webcam index (0 = default webcam)
    else:
        input_stream = "your RTSP url"

    print("Loading YOLO...")
    yolo_model = load_yolo_model()
    print("Loading InsightFace...")
    arcface_app = load_arcface_model()
    print("Loading embeddings from MongoDB...")
    db_embeds = fetch_embedding_database_from_mongodb()

    print("Starting video processing...")
    process_video_live(input_stream, yolo_model, arcface_app, db_embeds, threshold=0.45)
