import streamlit as st
import keras
import model
import time

# Page configuration
st.set_page_config(
    page_title="SARCA-SENSE | Sarcasm Detection",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, minimalistic design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 1rem 0.5rem;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.3rem;
    }

    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .stats-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }

    .stats-label {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.3rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem 0;
        color: #888;
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Reduce padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Section headers */
    h3 {
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.3rem !important;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }

    /* Compact text area */
    .stTextArea>div>div>textarea {
        min-height: 80px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'sarcastic_count' not in st.session_state:
    st.session_state.sarcastic_count = 0
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Creator attribution
st.markdown("""
    <div style="text-align: center; padding: 0.3rem 0 0.8rem 0;">
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            <p>Created with ❤️ by <a href="https://shad-datascience.github.io" target="_blank" 
           style="color: #667eea; text-decoration: none; font-weight: 600;">Shad Jamil</a></p>
        </p>
    </div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🎭 SARCA-SENSE</h1>
        <p class="subtitle">AI-Powered Hinglish Sarcasm Detection</p>
    </div>
""", unsafe_allow_html=True)


# Load model (cached)
@st.cache_resource
def load_classifier():
    try:
        return keras.models.load_model('./src/mlp_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


classifier = load_classifier()


# Function to perform prediction
def perform_prediction(input_text):
    if classifier is None:
        st.error("❌ Model could not be loaded. Please check the model file path.")
        return None

    try:
        transformed_text = model.feature_pipeline.transform([input_text])
        prediction = classifier.predict(transformed_text)
        score = float(prediction[0][0])

        # Update statistics
        st.session_state.total_predictions += 1
        is_sarcastic = score >= 0.5

        if is_sarcastic:
            st.session_state.sarcastic_count += 1

        # Store in history
        st.session_state.prediction_history.insert(0, {
            'text': input_text[:50] + '...' if len(input_text) > 50 else input_text,
            'is_sarcastic': is_sarcastic,
            'score': score
        })

        # Keep only last 5 predictions
        st.session_state.prediction_history = st.session_state.prediction_history[:5]

        result = {'is_sarcastic': is_sarcastic, 'score': score}
        st.session_state.last_prediction = result
        return result

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please ensure the model and feature pipeline are properly configured.")
        return None


# Main input section
st.markdown("### 📝 Enter Your Text")

text_input = st.text_area(
    "Type or paste your Hinglish sentence:",
    value=st.session_state.current_text,
    height=80,
    placeholder="Hint: try longer sentences for better response ...",
    key="main_text_area",
    label_visibility="collapsed"
)

# Predict button
predict_button = st.button("🔍 Analyze Sarcasm", use_container_width=True, type="primary")

# Example sentences
st.markdown("### 💡 Try Examples")

examples = [
    "Wah Bete mauj kardi, tum to bade heavy driver ho bhai",
    "Aaaj Mausam bada accha hua hai, chai peene ka man kar rha aaj to",
    "Bhai tu to Einstein se bhi tej hai, tuje to 2+2 ke liye bhi calculator lag rha hai",
    "Chal Bhai kahi bahar ghumne chalte hai bahut din hogaye kahi gaye huye"
]

cols = st.columns(2)
example_clicked = None

for idx, example in enumerate(examples):
    with cols[idx % 2]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            example_clicked = example

# Handle example click - update text and predict immediately
if example_clicked:
    st.session_state.current_text = example_clicked
    with st.spinner("🤔 Analyzing..."):
        time.sleep(0.3)
        result = perform_prediction(example_clicked)
    st.rerun()

# Handle manual prediction button
if predict_button:
    if text_input and text_input.strip():
        st.session_state.current_text = text_input
        with st.spinner("🤔 Analyzing..."):
            time.sleep(0.3)
            result = perform_prediction(text_input)
            st.rerun()
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Display last prediction result
if st.session_state.last_prediction:
    result = st.session_state.last_prediction

    st.markdown("### 🎯 Result")

    col1, col2 = st.columns(2)

    with col1:
        if result['is_sarcastic']:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center;'>
                    <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🌀 Sarcastic</h2>
                    <p style='color: rgba(255,255,255,0.9); margin-top: 0.3rem; font-size: 0.9rem;'>
                        Sarcasm detected!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center;'>
                    <h2 style='color: white; margin: 0; font-size: 1.5rem;'>✅ Not Sarcastic</h2>
                    <p style='color: rgba(255,255,255,0.9); margin-top: 0.3rem; font-size: 0.9rem;'>
                        No sarcasm detected
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        confidence = result['score'] if result['is_sarcastic'] else (1 - result['score'])
        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.1f}%",
            delta=f"Raw: {result['score']:.4f}"
        )
        st.progress(confidence)

# Statistics section
if st.session_state.total_predictions > 0:
    st.markdown("### 📊 Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.total_predictions}</div>
                <div class="stats-label">Total</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.sarcastic_count}</div>
                <div class="stats-label">Sarcastic</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        non_sarcastic = st.session_state.total_predictions - st.session_state.sarcastic_count
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{non_sarcastic}</div>
                <div class="stats-label">Not Sarcastic</div>
            </div>
        """, unsafe_allow_html=True)

# Recent predictions
if st.session_state.prediction_history:
    st.markdown("### 📜 Recent")
    for pred in st.session_state.prediction_history[:3]:  # Show only 3
        emoji = "🌀" if pred['is_sarcastic'] else "✅"
        label = "Sarcastic" if pred['is_sarcastic'] else "Not Sarcastic"
        st.markdown(f"""
            <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 8px; 
                        margin: 0.4rem 0; border-left: 3px solid {"#f5576c" if pred['is_sarcastic'] else "#00f2fe"}'>
                <strong style='font-size: 0.9rem;'>{emoji} {label}</strong> 
                <span style='color: #999; font-size: 0.85rem;'>({pred['score']:.2f})</span><br>
                <span style='color: #666; font-size: 0.85rem;'>{pred['text']}</span>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p style="font-size: 0.75rem; color: #aaa; margin-top: 0.3rem;">
            Neural Networks • Transformers • Deep Learning • Sentiment Analysis • NLP
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    **SARCA-SENSE** Detects Sarcasm in Hinglish text using AI.

    **Features:**
    - Real-time detection
    - Confidence scoring
    - Session statistics
    - Prediction history
    """)

    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. Enter text or click example
    2. Get instant results
    3. View confidence scores
    """)

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.total_predictions = 0
        st.session_state.sarcastic_count = 0
        st.session_state.current_text = ""
        st.session_state.last_prediction = None
        st.rerun()

    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - Python
    - Streamlit
    - Keras/TensorFlow
    - Transformers
    - Numpy
    - Pandas
    - Scikit-learn
    """)
