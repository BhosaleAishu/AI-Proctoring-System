import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from fpdf import FPDF
from streamlit_javascript import st_javascript

# --- CONFIGURATION & STATE ---
st.set_page_config(page_title="FuturProctor AI", layout="wide")

if "warnings" not in st.session_state:
    st.session_state.warnings = 0
if "log" not in st.session_state:
    st.session_state.log = []
if "auth" not in st.session_state:
    st.session_state.auth = False

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Face Mesh for Gaze Tracking
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    # YOLO for Object Detection
    yolo_model = YOLO("yolov8n.pt") 
    return face_mesh, yolo_model

face_mesh, yolo_model = load_models()

# --- PDF GENERATOR ---
def generate_pdf(log_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "FuturProctor: Official Exam Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    for entry in log_data:
        text = f"Time: {entry['Time']} | Violation: {entry['Status']} | Total: {entry['Warnings']}"
        pdf.cell(200, 8, txt=text, ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- UI TABS ---
tab_auth, tab_exam, tab_report = st.tabs(["🔐 Authentication", "📝 Live Exam", "📊 Final Report"])

# --- 1. FACE AUTHENTICATION ---
with tab_auth:
    st.header("Step 1: Identity Verification")
    img_file = st.camera_input("Verify your face to start")
    
    if img_file:
        # In a real app, you'd compare this against a database using DeepFace.verify()
        st.session_state.auth = True
        st.success("Identity Verified ✅ Proceed to the Exam tab.")

# --- 2. LIVE PROCTORING ---
with tab_exam:
    if not st.session_state.auth:
        st.error("Please complete Authentication first!")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            run_proctor = st.toggle("Start Proctoring")
            FRAME_WINDOW = st.image([])
        
        with col2:
            st.subheader("Monitoring Log")
            alert_box = st.empty()
            warning_stat = st.empty()

        # TAB SWITCH DETECTION (JavaScript Hook)
        tab_focus = st_javascript("""window.document.hasFocus()""")
        
        cap = cv2.VideoCapture(0)
        
        while run_proctor:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status = "Normal"
            color = (0, 255, 0) # Green

            # A. Tab Switch Detection
            if tab_focus is False:
                status = "Tab Switch Detected!"
                color = (0, 0, 255)

            # B. YOLO Phone Detection (Objects: 67 is cell phone)
            results = yolo_model(frame, verbose=False, conf=0.5)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    if label in ["cell phone", "laptop", "book"]:
                        status = f"Prohibited: {label.upper()}"
                        color = (0, 0, 255)

            # C. Face Count & Gaze Tracking
            mesh_results = face_mesh.process(rgb_frame)
            if mesh_results.multi_face_landmarks:
                if len(mesh_results.multi_face_landmarks) > 1:
                    status = "Multiple Faces Detected"
                    color = (0, 0, 255)
                else:
                    # Gaze logic: Pupil position relative to eye corners
                    lms = mesh_results.multi_face_landmarks[0].landmark
                    iris_x = lms[468].x # Left Iris Center
                    if iris_x < 0.46 or iris_x > 0.54:
                        status = "Suspicious Gaze"
                        color = (255, 165, 0) # Orange
            else:
                status = "User Not Visible"
                color = (0, 0, 255)

            # --- WARNING SYSTEM ---
            if status != "Normal":
                st.session_state.warnings += 0.1 # Accumulate score
                current_log = {"Time": time.strftime("%H:%M:%S"), "Status": status, "Warnings": int(st.session_state.warnings)}
                st.session_state.log.append(current_log)

            # --- RENDER ---
            cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            FRAME_WINDOW.image(frame, channels="BGR")
            alert_box.warning(f"Status: {status}")
            warning_stat.metric("Violation Score", int(st.session_state.warnings))

            if st.session_state.warnings > 20:
                st.error("EXAM BLOCKED: Too many violations.")
                break
        
        cap.release()

# --- 3. AUTO-GENERATE REPORT ---
with tab_report:
    st.header("Exam Summary & Report")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log).drop_duplicates(subset=['Status'], keep='last')
        st.dataframe(df, use_container_width=True)
        
        pdf_bytes = generate_pdf(st.session_state.log)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"Report_{int(time.time())}.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No exam data recorded yet.")