import streamlit as st
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import threading
import time
import queue
from PIL import Image
import io
import tempfile

# Import your existing functions
from face_detection_recognitaion_rtsp import (
    load_yolo_model, 
    load_arcface_model, 
    fetch_embedding_database_from_mongodb,
    enhance_image,
    match_face,
    is_front_facing,
    is_attendance_marked_today,
    mark_attendance,
    log_access
)

from create_face_embeddings_yolo_arcface import (
    load_insightface_model,
    process_image_and_get_embedding,
    save_faiss_index,
    insert_to_mongodb as insert_embeddings_to_mongodb
)

# === MongoDB Connection ===
mongo_client = MongoClient("your mongob string")
mongo_db = mongo_client["face_db"]
attendance_collection = mongo_db["attendance_logs"]
embedding_collection = mongo_db["face_vectors_raw"]

# === Global Variables ===
if 'video_running' not in st.session_state:
    st.session_state.video_running = False
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None

# === Video Processing Thread ===
class VideoProcessor:
    def __init__(self, rtsp_url, yolo_model, arcface_app, db_embeds, threshold=0.45):
        self.rtsp_url = rtsp_url
        self.yolo_model = yolo_model
        self.arcface_app = arcface_app
        self.db_embeds = db_embeds
        self.threshold = threshold
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.marked_today = set()  # Thread-safe local variable
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_video, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _process_video(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            st.error("Failed to open video stream")
            return
            
        frame_idx = 0
        trackers, identities = [], []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            enhanced = enhance_image(frame)
            
            if frame_idx % 5 == 0:
                results = self.yolo_model(enhanced)
                boxes = results[0].boxes
                faces = self.arcface_app.get(enhanced)
                
                trackers.clear()
                identities.clear()
                
                if boxes and faces:
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
                            
                        emb = best_face.embedding
                        emb /= np.linalg.norm(emb)
                        name, sim = match_face(emb, self.db_embeds, self.threshold)
                        
                        if name != "Unknown":
                            if name not in self.marked_today:
                                if not is_attendance_marked_today(name):
                                    mark_attendance(name, sim)
                                    self.marked_today.add(name)
                            log_access(name, sim, "GRANTED")
                        else:
                            log_access("Unknown", sim, "DENIED")
                            
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
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} ({sim:.2f})" if name != "Unknown" else "Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            # Put frame in queue for display
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
                
            frame_idx += 1
            
        cap.release()

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
    st.title("Face Recognition Attendance")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Video Source Selection
        video_source = st.radio("Select Video Source", ["RTSP Camera", "Webcam"])
        
        if video_source == "RTSP Camera":
            rtsp_url = st.text_input("RTSP URL", "Your RTSP url")
            input_source = rtsp_url
        else:
            input_source = 0  # Webcam
            st.info("Using default webcam (index 0)")
        
        # Minimum similarity score (0 to 1) required to consider a face match as valid.
        threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        page = st.radio("Navigation", ["Live Attendance", "Enroll New Person", "View Reports", "Manage Database"])
    
    # PAGE 1: Live Attendance 
    if page == "Live Attendance":
        st.header("Live Attendance Monitoring")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Video Feed")
            video_placeholder = st.empty()
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_btn = st.button("Start Monitoring", use_container_width=True)
            with col_btn2:
                stop_btn = st.button("Stop Monitoring", use_container_width=True)
                
            if start_btn and not st.session_state.video_running:
                with st.spinner("Loading models..."):
                    if 'yolo_model' not in st.session_state:
                        st.session_state.yolo_model = load_yolo_model()
                    if 'arcface_app' not in st.session_state:
                        st.session_state.arcface_app = load_arcface_model()
                    if 'db_embeds' not in st.session_state:
                        st.session_state.db_embeds = fetch_embedding_database_from_mongodb()
                        
                st.session_state.processor = VideoProcessor(
                    input_source,  # Changed from rtsp_url to input_source
                    st.session_state.yolo_model,
                    st.session_state.arcface_app,
                    st.session_state.db_embeds,
                    threshold
                )
                st.session_state.processor.start()
                st.session_state.video_running = True
                st.success("Monitoring started!")
                
            if stop_btn and st.session_state.video_running:
                st.session_state.processor.stop()
                st.session_state.video_running = False
                st.info("Monitoring stopped")
                
            # Display video continuously
            if st.session_state.video_running and 'processor' in st.session_state:
                while st.session_state.video_running:
                    try:
                        frame = st.session_state.processor.frame_queue.get(timeout=0.1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    except queue.Empty:
                        time.sleep(0.01)
                        continue
                    except Exception as e:
                        break
                    
        with col2:
            st.subheader("Today's Attendance")
            attendance_placeholder = st.empty()
            
            # Get all registered users
            all_users = sorted(embedding_collection.distinct("name"))
            
            # Get today's attendance
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_data = list(attendance_collection.find({"date": today}))
            
            # Create a dictionary of present users
            present_users = {}
            for record in attendance_data:
                present_users[record['name']] = {
                    'arrival_time': record['arrival_time'],
                    'similarity': record['similarity']
                }
            
            # Build complete attendance list
            attendance_list = []
            for user in all_users:
                if user in present_users:
                    attendance_list.append({
                        'Name': user,
                        'Arrival Time': present_users[user]['arrival_time'],
                        'Confidence': f"{present_users[user]['similarity']:.2f}",
                        'Status': 'Present'
                    })
                else:
                    attendance_list.append({
                        'Name': user,
                        'Arrival Time': '-',
                        'Confidence': '-',
                        'Status': 'Absent'
                    })
            
            if attendance_list:
                df = pd.DataFrame(attendance_list)
                #attendance_placeholder.dataframe(df, use_container_width=True, hide_index=True)
                def color_status(val):
                    if val == 'Present':
                        return 'background-color: #006400'  # Dark green
                    elif val == 'Absent':
                        return 'background-color: #8B0000'  # Dark red
                    return ''

                styled_df = df.style.applymap(color_status, subset=['Status'])
                attendance_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Summary metrics
                present_count = len([u for u in attendance_list if u['Status'] == 'Present'])
                absent_count = len(all_users) - present_count
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Present", present_count)
                col_m2.metric("Absent", absent_count)
                
                if present_count > 0:
                    avg_conf = sum([float(u['Confidence']) for u in attendance_list if u['Confidence'] != '-']) / present_count
                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
            else:
                attendance_placeholder.info("No registered users in database")
                
    # PAGE 2: Enroll New Person
    elif page == "Enroll New Person":
        st.header("Enroll New Person")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Enrollment Options")
            enroll_method = st.radio("Choose enrollment method:", 
                                    ["RTSP Stream", "Upload Images", "Upload Video"])
            
            person_name = st.text_input("Enter Person's Name", placeholder="Enter your name here")
            
            if enroll_method == "RTSP Stream":
                rtsp_enroll = st.text_input("RTSP URL for Enrollment", "Your RTSP url")
                duration = st.slider("Capture Duration (seconds)", 5, 30, 15)
                
                if st.button("Start Enrollment", use_container_width=True):
                    if person_name:
                        with st.spinner(f"Capturing frames for {person_name}..."):
                            from create_face_embeddings_yolo_arcface import recognize_and_record_new_entry
                            embeddings, metadata = recognize_and_record_new_entry(
                                rtsp_enroll, person_name, duration=duration
                            )
                            if embeddings:
                                st.success(f"Successfully enrolled {person_name} with {len(embeddings)} embeddings!")
                                # Reload embeddings
                                if 'db_embeds' in st.session_state:
                                    st.session_state.db_embeds = fetch_embedding_database_from_mongodb()
                            else:
                                st.error("No face detected. Please try again.")
                    else:
                        st.warning("Please enter a name first!")
                        
            elif enroll_method == "Upload Images":
                uploaded_files = st.file_uploader("Upload face images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
                
                if st.button("Save Images", use_container_width=True) and uploaded_files and person_name:
                    app = load_insightface_model()
                    embeddings, metadata = [], []
                    
                    progress_bar = st.progress(0)
                    for idx, file in enumerate(uploaded_files):
                        img = Image.open(file)
                        img_array = np.array(img)
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        emb = process_image_and_get_embedding(app, img_array)
                        if emb is not None:
                            embeddings.append(emb)
                            metadata.append(f"{person_name}/{file.name}")
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    if embeddings:
                        # Compute universal embedding
                        universal_embedding = np.mean(np.vstack(embeddings), axis=0)
                        universal_embedding = universal_embedding / np.linalg.norm(universal_embedding)
                        
                        save_faiss_index([universal_embedding], [f"{person_name}/universal_embedding"])
                        insert_embeddings_to_mongodb([f"{person_name}/universal_embedding"], [universal_embedding])
                        
                        st.success(f"Successfully enrolled {person_name} with {len(embeddings)} images!")
                        if 'db_embeds' in st.session_state:
                            st.session_state.db_embeds = fetch_embedding_database_from_mongodb()
                    else:
                        st.error("No faces detected in uploaded images!")
                        
            elif enroll_method == "Upload Video":
                uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'])
                
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    frame_interval = st.number_input("Frame Interval", min_value=5, max_value=30, value=15, 
                                                    help="Extract 1 frame every N frames")
                with col_v2:
                    max_frames = st.number_input("Max Frames", min_value=50, max_value=300, value=100,
                                                help="Maximum frames to process")
                
                if st.button("Process Video", use_container_width=True) and uploaded_video and person_name:
                    app = load_insightface_model()
                    embeddings, metadata = [], []
                    
                    # Save uploaded video to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        video_path = tmp_file.name
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        frame_count = 0
                        processed_count = 0
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while cap.isOpened() and processed_count < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if frame_count % frame_interval == 0:
                                emb = process_image_and_get_embedding(app, frame)
                                if emb is not None:
                                    embeddings.append(emb)
                                    metadata.append(f"{person_name}/{uploaded_video.name}_frame_{frame_count}")
                                    processed_count += 1
                                    status_text.text(f"Processing: Frame {frame_count} - Embeddings: {len(embeddings)}")
                                    progress_bar.progress(min(processed_count / max_frames, 1.0))
                            
                            frame_count += 1
                        
                        cap.release()
                        
                        if embeddings:
                            # Compute universal embedding
                            universal_embedding = np.mean(np.vstack(embeddings), axis=0)
                            universal_embedding = universal_embedding / np.linalg.norm(universal_embedding)
                            
                            save_faiss_index([universal_embedding], [f"{person_name}/universal_embedding"])
                            insert_embeddings_to_mongodb([f"{person_name}/universal_embedding"], [universal_embedding])
                            
                            st.success(f"Successfully enrolled {person_name} with {len(embeddings)} video frames!")
                            if 'db_embeds' in st.session_state:
                                st.session_state.db_embeds = fetch_embedding_database_from_mongodb()
                        else:
                            st.error("No faces detected in the video!")
                    
                    finally:
                        # Clean up temporary file
                        import os
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        
        with col2:
            st.subheader("Enrollment Instructions")
            st.info("""
            **For best results:**
            - Look directly at the camera
            - Ensure good lighting
            - Keep a neutral expression
            - Avoid sunglasses or masks
            - Stay still during capture
            
            **Video Upload Tips:**
            - 5-10 second videos work best
            - Move head slightly for different angles
            - Keep face clearly visible throughout
            """)
            
    # PAGE 3: View Reports
    elif page == "View Reports":
        st.header("Attendance Reports")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            date_filter = st.date_input("Select Date", datetime.now())
        with col2:
            search_name = st.text_input("Search by Name", "")
            
        # Fetch data
        date_str = date_filter.strftime("%Y-%m-%d")
        query = {"date": date_str}
        if search_name:
            query["name"] = {"$regex": search_name, "$options": "i"}
            
        records = list(attendance_collection.find(query).sort("arrival_time", 1))
        
        if records:
            df = pd.DataFrame(records)
            df = df[['name', 'arrival_time', 'similarity', 'status']]
            df.columns = ['Name', 'Arrival Time', 'Confidence', 'Status']
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"attendance_{date_str}.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Present", len(df))
            col2.metric("Unique People", df['Name'].nunique())
            col3.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")
        else:
            st.info("No records found for selected criteria")
            
    # PAGE 4: Manage Database
    elif page == "Manage Database":
        st.header("Database Management")
        
        # Show registered users
        st.subheader("Registered Users")
        users = embedding_collection.distinct("name")
        
        if users:
            user_counts = {}
            for user in users:
                count = embedding_collection.count_documents({"name": user})
                user_counts[user] = count
                
            df_users = pd.DataFrame(list(user_counts.items()), columns=['Name', 'Embeddings Count'])
            st.dataframe(df_users, use_container_width=True, hide_index=True)
            
            st.metric("Total Registered Users", len(users))
            
            # Delete user
            st.subheader("Delete User")
            user_to_delete = st.selectbox("Select user to delete", [""] + users)
            if user_to_delete and st.button("Delete User", type="secondary"):
                if st.checkbox("I confirm deletion"):
                    embedding_collection.delete_many({"name": user_to_delete})
                    attendance_collection.delete_many({"name": user_to_delete})
                    st.success(f"Deleted {user_to_delete} from database")
                    st.rerun()
        else:
            st.info("No users registered yet")
            
        # Clear attendance logs
        st.subheader("Clear Attendance Logs")
        if st.button("Clear All Attendance Records", type="secondary"):
            if st.checkbox("I confirm clearing all records"):
                attendance_collection.delete_many({})
                st.success("All attendance records cleared")
                st.rerun()

if __name__ == "__main__":
    main()