import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🌍 Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔽 Downloading model... (only once)"):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                downloaded = 0
                
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(downloaded / total_size)
                
                st.success("✅ Model downloaded successfully!")
                return load_model(MODEL_PATH)
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                return None
    else:
        return load_model(MODEL_PATH)

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Model info
    st.subheader("📊 Model Information")
    st.info("""
    **Model**: Satellite Image Classifier v1
    **Input Size**: 256x256 pixels
    **Classes**: 4 categories
    **Framework**: TensorFlow/Keras
    """)
    
    # Settings
    st.subheader("⚙️ Settings")
    show_confidence_chart = st.checkbox("Show confidence chart", value=True)
    show_all_predictions = st.checkbox("Show all class probabilities", value=True)
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - Use clear satellite images
    - Best results with 256x256 resolution
    - Avoid heavily processed images
    - Good lighting conditions work best
    """)

# Main content
st.title("🌍 Satellite Image Classifier")
st.markdown("### Upload a satellite image to classify it into one of four categories")

# Class information
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("☁️ Cloudy", "Weather", help="Areas covered by clouds")
with col2:
    st.metric("🏜️ Desert", "Terrain", help="Arid, sandy regions")
with col3:
    st.metric("🌿 Green Area", "Vegetation", help="Forests, agricultural land")
with col4:
    st.metric("💧 Water", "Bodies", help="Lakes, rivers, oceans")

st.markdown("---")

# Load model
model = download_and_load_model()

if model is None:
    st.error("❌ Failed to load model. Please check your internet connection and try again.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "📁 Choose a satellite image file",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG (max 200MB)"
)

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Input Image")
        
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        image_resized = image.resize((256, 256))
        
        st.image(image_resized, caption="Uploaded Image (resized to 256x256)", use_container_width=True)
        
        # Image details
        st.info(f"""
        **Original Size**: {original_size[0]} × {original_size[1]}
        **File Size**: {len(uploaded_file.getvalue())} bytes
        **Format**: {uploaded_file.type}
        """)
    
    with col2:
        st.subheader("🎯 Classification Results")
        
        # Preprocess image
        img_array = img_to_array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class_idx = np.argmax(prediction)
            predicted_class = ['Cloudy', 'Desert', 'Green_Area', 'Water'][predicted_class_idx]
            confidence = np.max(prediction)
            
            # Display main result
            emoji_map = {'Cloudy': '☁️', 'Desert': '🏜️', 'Green_Area': '🌿', 'Water': '💧'}
            
            st.success(f"**Prediction**: {emoji_map[predicted_class]} {predicted_class}")
            
            # Confidence meter
            confidence_pct = confidence * 100
            if confidence_pct >= 80:
                st.success(f"**Confidence**: {confidence_pct:.1f}% - High")
            elif confidence_pct >= 60:
                st.warning(f"**Confidence**: {confidence_pct:.1f}% - Medium")
            else:
                st.error(f"**Confidence**: {confidence_pct:.1f}% - Low")
            
            # Progress bar for confidence
            st.progress(confidence)
    
    # Additional visualizations
    st.markdown("---")
    
    if show_confidence_chart:
        st.subheader("📈 Confidence Distribution")
        
        # Create confidence chart
        class_names = ['Cloudy', 'Desert', 'Green Area', 'Water']
        colors = ['#87CEEB', '#DEB887', '#228B22', '#4169E1']
        
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=prediction * 100,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in prediction * 100],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence for Each Class",
            xaxis_title="Class",
            yaxis_title="Confidence (%)",
            showlegend=False,
            height=400
        )
        
        # Highlight the predicted class
        fig.add_shape(
            type="rect",
            x0=predicted_class_idx - 0.4,
            y0=0,
            x1=predicted_class_idx + 0.4,
            y1=prediction[predicted_class_idx] * 100,
            fillcolor="rgba(255,255,0,0.3)",
            line=dict(color="gold", width=2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if show_all_predictions:
        st.subheader("📊 All Class Probabilities")
        
        # Create a detailed table
        results_df = []
        for i, (class_name, prob) in enumerate(zip(class_names, prediction)):
            emoji = ['☁️', '🏜️', '🌿', '💧'][i]
            is_predicted = i == predicted_class_idx
            results_df.append({
                'Class': f"{emoji} {class_name}",
                'Probability': f"{prob:.4f}",
                'Percentage': f"{prob*100:.2f}%",
                'Predicted': "✅" if is_predicted else ""
            })
        
        st.dataframe(results_df, use_container_width=True)
    
    # Export results
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    # Create export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'filename': uploaded_file.name,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'all_probabilities': {
            'Cloudy': float(prediction[0]),
            'Desert': float(prediction[1]),
            'Green_Area': float(prediction[2]),
            'Water': float(prediction[3])
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Results (JSON)",
            data=str(export_data),
            file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Create a summary text
        summary_text = f"""
Satellite Image Classification Results
====================================

File: {uploaded_file.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTION: {emoji_map[predicted_class]} {predicted_class}
CONFIDENCE: {confidence_pct:.2f}%

Detailed Probabilities:
- Cloudy: {prediction[0]*100:.2f}%
- Desert: {prediction[1]*100:.2f}%
- Green Area: {prediction[2]*100:.2f}%
- Water: {prediction[3]*100:.2f}%
        """
        
        st.download_button(
            label="📝 Download Summary (TXT)",
            data=summary_text,
            file_name=f"classification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h3>👆 Upload a satellite image to get started!</h3>
        <p>The AI will analyze your image and classify it into one of four categories:</p>
        <p><strong>☁️ Cloudy • 🏜️ Desert • 🌿 Green Area • 💧 Water</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌍 Satellite Image Classifier | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
