import streamlit as st
from PIL import Image
from model import PlateDetector

# Set up page configurations
st.set_page_config(page_title="AI License Plate Detector", page_icon="🚗", layout="centered")

st.title("🚗 AI License Plate Detector")
st.write("Upload an image of a vehicle below, and our YOLO model will attempt to locate the license plate.")

# Initialize the model using Streamlit's session state cache 
# This ensures the model only loads ONCE when the app starts up, making it fast.
@st.cache_resource
def load_cached_detector():
    with st.spinner("Downloading and initializing YOLO model weights from Kaggle... Please wait."):
        return PlateDetector()

try:
    detector = load_cached_detector()
except Exception as e:
    st.error(f"Failed to load model weights: {e}")
    st.stop()

st.divider()

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI Detection Output")
        with st.spinner("Analyzing image..."):
            # Run prediction
            output_image, raw_results = detector.predict(image)
            
            # Display results
            st.image(output_image, use_container_width=True)
            
   # Display statistics underneath
    num_plates = len(raw_results.boxes)
    if num_plates > 0:
        st.success(f"🎉 Successfully detected {num_plates} license plate(s)!")
        
        # Show raw bounding box telemetry data for developers
        with st.expander("See raw bounding box data"):
            for box in raw_results.boxes:
                st.write(f"Confidence: {box.conf[0]:.2f}")
                st.write(f"Coordinates [xyxy]: {box.xyxy[0]}")
    else:
        st.warning("No license plates detected. Try an image with a clearer angle or better lighting.")