import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import os
import cv2
from moviepy.editor import VideoFileClip

# Page Config
st.set_page_config(page_title="AI Content Detector", layout="wide")

# Load Models
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="roberta-base-openai-detector")  # Model for text detection

@st.cache_resource
def load_image_model():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")  # Model for image detection

# Utility Functions
def analyze_text(text):
    model = load_text_model()
    result = model(text)
    return result

def analyze_image(image):
    model = load_image_model()
    result = model(image)
    return result

def extract_frames_from_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = fps // frame_rate
    success, frame = cap.read()
    count = 0
    while success:
        if count % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames

def analyze_video(video_path):
    frames = extract_frames_from_video(video_path, frame_rate=1)
    results = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        result = analyze_image(pil_image)
        results.append(result)
    return results

# Streamlit App
st.title("AI Content Detector")
st.write("Detect if content (text, image, or video) is AI-generated with probability scores.")

st.sidebar.write("### Authors")
st.sidebar.markdown("[Mahmoud Boghdady](https://www.linkedin.com/in/mahmoud-boghdady-msc-pmp%C2%AE-6b694033/)")
st.sidebar.markdown("[Abu Bakar Rasheed](https://www.linkedin.com/in/abu-bakar-rasheed-9b65b616/)")
st.sidebar.write("### Responsible AI")
st.sidebar.write(
    "Responsible AI is a framework for ensuring that AI systems are developed and deployed in a manner that is fair, transparent, and aligned with societal values. "
    "It focuses on accountability, mitigating biases, and fostering trust in AI-driven solutions."
)

# Sidebar Input
content_type = st.sidebar.selectbox("Select Content Type", ["Text", "Image", "Video"])

if content_type == "Text":
    st.header("Analyze Text")
    user_text = st.text_area("Enter text to analyze", "")
    if st.button("Analyze Text"):
        if user_text:
            with st.spinner("Analyzing..."):
                text_result = analyze_text(user_text)
            st.success("Analysis Complete")
            st.write("### Results:")
            st.json(text_result)
        else:
            st.warning("Please enter text to analyze.")

elif content_type == "Image":
    st.header("Analyze Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if st.button("Analyze Image"):
        if uploaded_image:
            with st.spinner("Analyzing..."):
                image = Image.open(uploaded_image)
                image_result = analyze_image(image)
            st.success("Analysis Complete")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("### Results:")
            st.json(image_result)
        else:
            st.warning("Please upload an image to analyze.")

elif content_type == "Video":
    st.header("Analyze Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if st.button("Analyze Video"):
        if uploaded_video:
            with st.spinner("Analyzing..."):
                video_path = f"temp_{uploaded_video.name}"
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.read())
                video_results = analyze_video(video_path)
                os.remove(video_path)
            st.success("Analysis Complete")
            st.write("### Results (Per Frame):")
            for i, frame_result in enumerate(video_results):
                st.write(f"Frame {i+1}: {frame_result}")
        else:
            st.warning("Please upload a video to analyze.")
