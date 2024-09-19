# Importing necessary libraries

# Audio processing and conversion
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import wave
import os

# Speech recognition and sentiment analysis
from google.cloud import speech_v1p1beta1 as speech
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import speech_recognition as sr

# PyTorch and audio libraries
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# NLP, topic modeling, and text processing
import spacy
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Diarization and language detection
from pyannote.audio import Pipeline
from langdetect import detect

# Utilities
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set NLTK data path (if necessary, e.g., custom location)
# You can append a custom path like this if you face issues with data loading:
nltk.data.path.append('')

# Pre-trained models and token loading
#os.environ["HF_TOKEN"] = "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"

# Load spacy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Load pre-trained Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load pre-trained speaker diarization pipeline with the token
#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.environ["HF_TOKEN"])


# ============================== AUDIO PROCESSING FUNCTIONS ==============================

# Function to check if a file is a valid WAV format
def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'r') as file:
            return True
    except wave.Error:
        return False

# Function to extract audio from a video file
def extract_audio_from_video(video_file: str):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile("extracted_audio.wav", codec="pcm_s16le")

# Function to convert audio files to WAV format (if necessary)
def convert_to_wav(file_path):
    if not file_path.endswith('.wav'):
        audio = AudioSegment.from_file(file_path)
        wav_file = file_path.rsplit('.', 1)[0] + ".wav"
        audio.export(wav_file, format="wav", codec="pcm_s16le")
        return wav_file
    return file_path


# ============================== SPEECH RECOGNITION FUNCTIONS ==============================

# Function to transcribe audio using Wav2Vec 2.0
def transcribe_audio(file_path):
    file_path = convert_to_wav(file_path)  # Convert to WAV if necessary

    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample to 16 kHz (if necessary)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Tokenize input
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

    # Perform transcription
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the transcription
    transcription = processor.decode(predicted_ids[0])
    return transcription


# ============================== TEXT ANALYSIS FUNCTIONS ==============================

# Function to analyze sentiments from text
def analyze_sentiments(text: str):
    if text is None:
        return "No text available for sentiment analysis."
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment


# ============================== TOPIC MODELING FUNCTIONS ==============================

# Preprocessing function for topic modeling
def preprocess_text(text):
    # Tokenize the text
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Lemmatize using spacy
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    return tokens

# Function to extract topics using LDA
def extract_topics(text, num_topics=3, num_words=5):
    tokens = preprocess_text(text)  # Preprocess the text

    # Create a dictionary representation of the documents
    id2word = corpora.Dictionary([tokens])

    # Create the Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(tokens)]

    # Build the LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=10,
                         passes=10,
                         alpha='auto')

    # Get the topics
    topics = lda_model.print_topics(num_words=num_words)
    return topics



# ============================== UTILITY FUNCTIONS ==============================

# Function to detect language of a given text
def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "Unable to detect language"

# Function to count words in a given text
def count_words(text):
    return len(text.split())

# Function to get the frequency of specified keywords in a text
def keyword_frequency(text, keywords):
    words = nltk.word_tokenize(text.lower())
    word_count = Counter(words)
    
    keyword_freq = {keyword: word_count[keyword] for keyword in keywords}
    return keyword_freq




