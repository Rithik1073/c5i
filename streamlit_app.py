import streamlit as st
import os
import sys
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Add project path
sys.path.append('/content/my_project')

# Import custom functions
from models import transcribe_audio, extract_audio_from_video, analyze_sentiments
from models import extract_topics, detect_language, count_words, keyword_frequency

# Load GPT-Neo model and tokenizer
try:
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Directory to save the uploaded files
UPLOAD_FOLDER = '/content/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Convert text to proper case
def convert_to_proper_case(text):
    return text.capitalize()

def generate_text(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit app
st.set_page_config(page_title="Conversational Insights Platform", page_icon="📊", layout="wide")

# Custom CSS for dark theme and neon green colors
st.markdown("""
    <style>
    body, .css-1v3fvcr, .css-18e3th9 {
        background-color: #000000; /* Black background */
        color: #e0e0e0; /* Light text color */
    }
    h1, h2, h3 {
        color: #39ff14; /* Neon green */
    }
    .stButton>button {
        background-color: #39ff14; /* Neon green */
        color: #ffffff;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #333333;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Conversational Insights Platform")

st.sidebar.title("Options")
st.sidebar.subheader("Upload and Analyze")
st.sidebar.write("Upload an audio or video file to get insights.")

# Upload file widget
uploaded_file = st.sidebar.file_uploader("Upload Audio or Video", type=["mp3", "wav", "mp4"])

if uploaded_file:
    with st.spinner('Processing the uploaded file...'):
        # Save file to the uploads folder
        file_path = save_uploaded_file(uploaded_file)

        # Process the uploaded file
        if file_path.endswith('.mp4'):
            st.info("Extracting audio from video...")
            extract_audio_from_video(file_path)
            audio_file = 'extracted_audio.wav'
        else:
            audio_file = file_path

        # Transcribe the audio (removed the info widget here)
        transcription = transcribe_audio(audio_file)
        transcription = convert_to_proper_case(transcription)  # Convert transcription to proper case

        # Display the transcription
        st.subheader("Transcription")
        st.text_area("Transcription Text", transcription, height=300)

        # Topic extraction
        st.subheader("Main Topics")
        topics = extract_topics(transcription, num_topics=3)
        for topic in topics:
            words = ', '.join([word.split('*')[1].strip().strip('"') for word in topic[1].split(' + ')])
            st.write(words)

        # Analyze sentiments of the transcription
        st.subheader("Sentiment Analysis")
        sentiment = analyze_sentiments(transcription)
        st.write(sentiment)

        # Summarize the transcription using GPT-Neo
        st.subheader("Summary")
        summary_prompt = f"Summarize the following conversation:\n{transcription}\nSummary:"
        summary = generate_text(summary_prompt, max_length=150)
        st.write("Summary:", summary)

        # Generate insights from the transcription using GPT-Neo
        st.subheader("Insights")
        insights_prompt = f"Analyze the following conversation and provide business insights:\n{transcription}\nInsights:"
        insights = generate_text(insights_prompt, max_length=150)
        st.write("Insights:", insights)

        # Language detection
        st.subheader("Language Detected")
        language = detect_language(transcription).capitalize()
        st.write(language)

        # Word count
        st.subheader("Number of Words Spoken")
        word_count = count_words(transcription)
        st.write(word_count)

        # Keyword frequency analysis
        st.subheader("Keyword Frequency")
        keywords = st.text_input("Enter keywords to analyze frequency (comma-separated)").split(',')
        if keywords:
            with st.spinner('Analyzing keyword frequency...'):
                keyword_freq = keyword_frequency(transcription, keywords)
                for keyword, freq in keyword_freq.items():
                    st.write(f"{keyword}: {freq}")

        # Ask questions about the transcription using GPT-Neo
        st.subheader("Ask Questions about the Audio/Video")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner('Generating answer...'):
                question_prompt = f"Conversation:\n{transcription}\nBased on this, {query}"
                answer = generate_text(question_prompt, max_length=100)
                st.write("Answer:", answer)
