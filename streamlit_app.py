# =========================
# IMPORTS
# =========================
import streamlit as st
import cv2
import numpy as np
import joblib
import torch

from facenet_pytorch import InceptionResnetV1, MTCNN

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("Welcome To Virtual Assistant")

# =========================
# SESSION STATE
# =========================
if "stage" not in st.session_state:
    st.session_state.stage = "face"

if "run" not in st.session_state:
    st.session_state.run = False

if "face_status" not in st.session_state:
    st.session_state.face_status = "stopped"

# =========================
# LOAD NLP MODELS
# =========================
intent_model = joblib.load("intent_model/model.pkl")
vectorizer = joblib.load("intent_model/vectorizer.pkl")
classes = joblib.load("intent_model/classes.pkl")

# =========================
# LOAD FACE MODELS
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

clf = joblib.load("face_svm.pkl")
le = joblib.load("label_encoder.pkl")

# =========================
# FACE EMBEDDING
# =========================
def get_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face)[0].cpu().numpy()

    emb = emb / np.linalg.norm(emb)
    return emb

# =========================
# FACE PREDICTION
# =========================
def predict_face(img):
    emb = get_embedding(img)

    if emb is None:
        return "No Face", 0

    pred = clf.predict([emb])[0]
    prob = clf.predict_proba([emb])[0].max()

    label = le.inverse_transform([pred])[0]

    if prob < 0.95:
        return "Unknown", prob

    return label, prob

# =========================
# NLP FUNCTION
# =========================
def predict_intent(text):
    vec = vectorizer.transform([text])
    prob = intent_model.predict_proba(vec)[0]

    label = classes[np.argmax(prob)]
    conf = np.max(prob)

    if conf < 0.90:
        return "UNKNOWN", conf

    return label, conf


# =========================
# STEP 1: FACE LOGIN
# =========================
if st.session_state.stage == "face":

    st.subheader("📷 Face Recognition (Login Required)")

    # STATUS
    if st.session_state.face_status == "running":
        st.success("Face Recognition: RUNNING 🟢")
    else:
        st.success("Face Recognition: STOPPED 🔴")

    col1, col2 = st.columns(2)

    # =========================
    # START BUTTON (GREEN WHEN ACTIVE)
    # =========================
    with col1:
        start_type = "primary" if st.session_state.face_status == "running" else "secondary"

        if st.button("▶ Start Face Recognition", type=start_type):
            st.session_state.run = True
            st.session_state.face_status = "running"
            st.rerun()

    # =========================
    # STOP BUTTON (GREEN WHEN ACTIVE)
    # =========================
    with col2:
        stop_type = "primary" if st.session_state.face_status == "stopped" else "secondary"

        if st.button("⛔ Stop Face Recognition", type=stop_type):
            st.session_state.run = False
            st.session_state.face_status = "stopped"
            st.rerun()

    # CAMERA INPUT
    img_file = st.camera_input("Take Snapshot")

    if st.session_state.run and img_file is not None:

        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:

            label, prob = predict_face(frame)

            if label == "ME":
                st.success("Access Granted ✅")
                st.session_state.stage = "nlp_unlock"
                st.session_state.run = False
                st.rerun()

            elif label == "No Face":
                text = "No Face Detected"
                color = (0, 0, 255)

            else:
                text = f"{label} ({prob:.2f})"
                color = (0, 0, 255)

            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            st.image(frame, channels="BGR")


# =========================
# STEP 2: UNLOCK SCREEN
# =========================
if st.session_state.stage == "nlp_unlock":

    st.subheader("🔓 Access Granted")
    st.success("You are verified as ME")

    if st.button("Enter AI Assistant (NLP)"):
        st.session_state.stage = "nlp"
        st.rerun()


# =========================
# STEP 3: NLP MODE
# =========================
if st.session_state.stage == "nlp":

    st.subheader("🧠 Intent Detection System")

    text = st.text_input("Enter command")

    if text:
        intent, conf = predict_intent(text)

        st.write("### Result")
        st.write("Text:", text)
        st.write("Intent:", intent)
        st.write("Confidence:", round(conf, 2))
