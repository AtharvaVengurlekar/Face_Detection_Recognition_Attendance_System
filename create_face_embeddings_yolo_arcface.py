import os
import cv2
import numpy as np
import faiss
import pickle
import time
from pathlib import Path
from collections import Counter
from pymongo import MongoClient
from insightface.app import FaceAnalysis

mongo_client = MongoClient("your Mongo DB String")
mongo_db = mongo_client["face_db"]
mongo_summary_col = mongo_db["face_embeddings_for_rtsp_testing"]
mongo_vector_col = mongo_db["face_vectors_raw"]

# === Load InsightFace ===
def load_insightface_model():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace initialized on GPU")
    except Exception as e:
        print(f"GPU init failed: {e}, switching to CPU")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace initialized on CPU")
    return app

def enhance_image(image):
    if image is None:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_brightness_contrast(image, alpha=1.1, beta=5):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def upscale_if_needed(image, min_size=112):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return image

def process_image_and_get_embedding(app, img):
    img = enhance_image(img)
    img = adjust_brightness_contrast(img)
    img = upscale_if_needed(img)
    faces = app.get(img)
    if not faces:
        return None
    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)

def extract_embeddings_from_dir(input_dir, frame_interval=15, max_frames=100):
    app = load_insightface_model()
    embeddings, metadata = [], []
    for person in os.listdir(input_dir):
        person_folder = os.path.join(input_dir, person)
        if not os.path.isdir(person_folder):
            continue
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(file_path)
                emb = process_image_and_get_embedding(app, img)
                if emb is not None:
                    embeddings.append(emb)
                    metadata.append(f"{person}/{file}")
                    print(f" Image: {person}/{file}")
            elif file.lower().endswith('.mp4'):
                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:
                        emb = process_image_and_get_embedding(app, frame)
                        if emb is not None:
                            embeddings.append(emb)
                            metadata.append(f"{person}/{file}_frame_{frame_count}")
                            print(f" Video Frame: {person}/{file}_frame_{frame_count}")
                    frame_count += 1
                cap.release()
    return embeddings, metadata

def recognize_and_record_new_entry(rtsp_url, person_name, duration=15, frame_interval=5):
    app = load_insightface_model()
    cap = cv2.VideoCapture(rtsp_url)
    embeddings, metadata = [], []

    print(f"\n Recording embeddings for: {person_name}")
    print(" Capturing frames from RTSP stream for 15 seconds...")

    start_time = time.time()
    while cap.isOpened() and time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        emb = process_image_and_get_embedding(app, frame)
        if emb is not None:
            embeddings.append(emb)
            metadata.append(f"{person_name}/rtsp_frame_{int(time.time())}")
            print(f" Saved embedding for {person_name}")
        time.sleep(1.0 / frame_interval)

    cap.release()
    print(f"\n Total new embeddings collected for {person_name}: {len(embeddings)}")

    if embeddings:
        # Compute universal embedding
        universal_embedding = np.mean(np.vstack(embeddings), axis=0)
        universal_embedding = universal_embedding / np.linalg.norm(universal_embedding)

        # Save single embedding to FAISS
        save_faiss_index([universal_embedding], [f"{person_name}/universal_embedding"], output_dir="FAISS_Vector_Embeddings")

        # Store in MongoDB
        insert_to_mongodb([f"{person_name}/universal_embedding"], [universal_embedding])

    else:
        print("No new embeddings saved.")

    return embeddings, metadata

def save_faiss_index(embeddings, metadata, output_dir="FAISS_Vector_Embeddings"):
    if not embeddings:
        print("No embeddings to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    dim = embeddings[0].shape[0]

    index_path = os.path.join(output_dir, "face_index.faiss")
    meta_path = os.path.join(output_dir, "face_index_ids.pkl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(" Existing FAISS index found. Appending new embeddings...")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            existing_metadata = pickle.load(f)

        index.add(np.vstack(embeddings).astype(np.float32))
        metadata = existing_metadata + metadata  # Append only new metadata

    else:
        print("No existing FAISS index found. Creating a new one...")
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(embeddings).astype(np.float32))

    # Save updated index and metadata
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f" FAISS index saved to '{output_dir}' with total {index.ntotal} embeddings.")
    insert_summary_to_mongodb(metadata)

def insert_summary_to_mongodb(metadata):
    items = [item for item in metadata if isinstance(item, str)]
    folders = [item.split("/")[0] for item in items]
    folder_counts = Counter(folders)

    for name, count in folder_counts.items():
        mongo_summary_col.update_one(
            {"name": name},
            {"$set": {"frame_count": count}},
            upsert=True
        )
        print(f"📊 {name}: {count} embeddings (upserted to MongoDB)")

def insert_to_mongodb(metadata, embeddings):
    documents = []
    for name, emb in zip(metadata, embeddings):
        documents.append({
            "name": name.split("/")[0],
            "source": name,
            "embedding": emb.tolist()
        })
    if documents:
        mongo_vector_col.insert_many(documents)
        print(f"\n {len(documents)} embeddings inserted into MongoDB collection: face_vectors_raw")

def main():
    print("Hello, this is a script to create face embeddings from different sources.")
    print("Please select the option on which the source would be used to create embeddings:")
    print("1. Path of the Frames")
    print("2. Path of the Videos")
    print("3. RTSP link")
    print(" NOTE: Please be in front of the camera and make sure to follow the instructions.\n")

    option = input("Enter option [1 / 2 / 3]: ").strip()

    if option in ["1", "2"]:
        path = input("Enter path to folder: ").strip()
        if not os.path.isdir(path):
            print("Invalid directory.")
            return
        embeddings, metadata = extract_embeddings_from_dir(path)
        save_faiss_index(embeddings, metadata, output_dir="FAISS_Vector_Embeddings")
        insert_to_mongodb(metadata, embeddings)

    elif option == "3":
        rtsp_url = input("Your RTSP url").strip()
        person_name = input("Please Enter the Name of the Person seen in the RTSP: ").strip()
        recognize_and_record_new_entry(rtsp_url, person_name)

    else:
        print(" Invalid option selected.")

if __name__ == "__main__":
    main()
