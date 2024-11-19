import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from datetime import datetime
from faster_whisper import WhisperModel
import torch
import os

class WhisperSpeechToText:
    def __init__(self, model_size="base", device="auto", compute_type="float32"):
        """
        Initialize the speech to text converter with Faster Whisper
        model_size options: "tiny", "base", "small", "medium", "large-v2"
        """
        # Automatically select device (CUDA if available, else CPU)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        
        # Audio recording parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1
        
    def record_audio(self, duration=5):
        """
        Record audio from microphone for specified duration
        Returns the recorded audio as numpy array
        """
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        sd.wait()  # Wait until recording is finished
        
        print("Recording finished!")
        return recording
    
    def save_audio(self, audio_data, filename=None):
        """
        Save the recorded audio to a WAV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        
        sf.write(filename, audio_data, self.sample_rate)
        return filename
    
    def transcribe_audio(self, audio_data=None, audio_file=None, language=None):
        """
        Transcribe audio using Faster Whisper
        Can process either audio data or an audio file
        """
        try:
            if audio_data is not None:
                # Save audio data to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    self.save_audio(audio_data, temp_file.name)
                    audio_path = temp_file.name
            else:
                audio_path = audio_file
            
            # Transcribe with Faster Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True
            )
            
            # Process segments
            transcription = []
            for segment in segments:
                transcription.append({
                    'text': segment.text,
                    'start': segment.start,
                    'end': segment.end,
                    'words': [{'word': word.word, 'start': word.start, 'end': word.end}
                             for word in segment.words]
                })
            
            return {
                'segments': transcription,
                'detected_language': info.language,
                'language_probability': info.language_probability
            }
            
        except Exception as e:
            return f"Transcription error: {str(e)}"
    
    def start_continuous_recognition(self, segment_duration=5, language=None):
        """
        Continuously record and transcribe audio in segments
        """
        print("Starting continuous recognition... (Press Ctrl+C to stop)")
        try:
            while True:
                # Record audio segment
                audio_data = self.record_audio(segment_duration)
                
                # Transcribe segment
                result = self.transcribe_audio(audio_data=audio_data, language=language)
                
                if isinstance(result, dict):
                    # Print transcription for each segment
                    for segment in result['segments']:
                        print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")
                else:
                    print(result)  # Print error message
                
        except KeyboardInterrupt:
            print("\nStopping continuous recognition...")

def main():
    # Create instance of speech to text converter
    print("Initializing Speech-to-Text system...")
    stt = WhisperSpeechToText(model_size="base")  # You can change model size here
    
    while True:
        print("\nSpeech to Text Options:")
        print("1. Record and transcribe")
        print("2. Transcribe existing audio file")
        print("3. Start continuous recognition")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            duration = int(input("Enter recording duration in seconds: "))
            audio_data = stt.record_audio(duration)
            filename = stt.save_audio(audio_data)
            print(f"Saved recording to {filename}")
            
            result = stt.transcribe_audio(audio_data=audio_data)
            if isinstance(result, dict):
                print("\nTranscription:")
                for segment in result['segments']:
                    print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")
                print(f"\nDetected language: {result['detected_language']} "
                      f"(probability: {result['language_probability']:.2f})")
            else:
                print(result)
            
        elif choice == '2':
            file_path = input("Enter the path to the audio file: ")
            if os.path.exists(file_path):
                result = stt.transcribe_audio(audio_file=file_path)
                if isinstance(result, dict):
                    print("\nTranscription:")
                    for segment in result['segments']:
                        print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")
                    print(f"\nDetected language: {result['detected_language']} "
                          f"(probability: {result['language_probability']:.2f})")
                else:
                    print(result)
            else:
                print("File not found!")
            
        elif choice == '3':
            language = input("Enter language code (leave empty for auto-detection): ").strip() or None
            stt.start_continuous_recognition(language=language)
            
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()