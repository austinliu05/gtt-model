import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import os
from pathlib import Path

class AudioToMIDIConverter:
    def __init__(self):
        """Initialize the audio-to-MIDI converter with Basic Pitch model."""
        self.model_path = ICASSP_2022_MODEL_PATH
        
    def extract_melody(self, audio_file_path, output_dir="temp_audio", remove_vocals=True):
        """
        Extract melody from audio file using harmonic-percussive separation and vocal removal.
        
        Args:
            audio_file_path (str): Path to input audio file
            output_dir (str): Directory to save processed audio
            remove_vocals (bool): Whether to attempt vocal removal
            
        Returns:
            str: Path to extracted melody audio file
        """
        print(f"Loading audio file: {audio_file_path}")
        y, sr = librosa.load(audio_file_path, sr=None)
        
        print("Separating harmonic (melody) from percussive components...")
        # Use HPSS to separate melody from drums/percussion
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        if remove_vocals:
            print("Attempting vocal removal...")
            # Try to remove vocals using center channel extraction
            y_no_vocals = self._remove_vocals_simple(y_harmonic, sr)
            
            # Further process to isolate instrumental melody
            y_instrumental = self._extract_instrumental_melody(y_no_vocals, sr)
        else:
            y_instrumental = y_harmonic
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed melody component
        melody_file_path = os.path.join(output_dir, "extracted_melody.wav")
        sf.write(melody_file_path, y_instrumental, sr)
        
        print(f"Melody extracted and saved to: {melody_file_path}")
        
        # Visualize the separation
        if remove_vocals:
            self._visualize_separation_with_vocals(y, y_harmonic, y_percussive, y_instrumental, sr)
        else:
            self._visualize_separation(y, y_harmonic, y_percussive, sr)
        
        return melody_file_path, sr
    
    def _remove_vocals_simple(self, y, sr):
        """
        Simple vocal removal using center channel extraction.
        This is a basic approach that works well for stereo recordings.
        """
        # If stereo, extract center channel (vocals are typically centered)
        if len(y.shape) > 1 and y.shape[0] == 2:
            # Convert to mono by taking difference (removes center-panned vocals)
            y_no_vocals = y[0] - y[1]
        else:
            # For mono audio, use spectral gating to reduce vocal frequencies
            y_no_vocals = self._spectral_vocal_reduction(y, sr)
        
        return y_no_vocals
    
    def _spectral_vocal_reduction(self, y, sr):
        """
        Reduce vocal frequencies using spectral gating.
        """
        # Get spectrogram
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Vocal frequencies are typically in the 85-255 Hz and 2-4 kHz range
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Create a mask to reduce vocal frequencies
        vocal_mask = np.ones_like(magnitude)
        
        # Reduce frequencies in vocal range (85-255 Hz and 2-4 kHz)
        vocal_freq_low = (freqs >= 85) & (freqs <= 255)
        vocal_freq_high = (freqs >= 2000) & (freqs <= 4000)
        
        vocal_mask[vocal_freq_low] *= 0.3  # Reduce low vocal frequencies
        vocal_mask[vocal_freq_high] *= 0.5  # Reduce high vocal frequencies
        
        # Apply mask
        magnitude_filtered = magnitude * vocal_mask
        
        # Reconstruct audio
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        y_filtered = librosa.istft(stft_filtered)
        
        return y_filtered
    
    def _extract_instrumental_melody(self, y, sr):
        """
        Further process audio to extract instrumental melody.
        """
        # Apply additional filtering to isolate melodic instruments
        # Remove very low frequencies (bass) and very high frequencies (cymbals)
        y_filtered = librosa.effects.preemphasis(y)
        
        # Apply spectral gating to emphasize melodic content
        stft = librosa.stft(y_filtered)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Emphasize mid-range frequencies where most melodic instruments live
        freqs = librosa.fft_frequencies(sr=sr)
        melody_mask = np.ones_like(magnitude)
        
        # Boost frequencies where melodic instruments typically live (200-2000 Hz)
        melody_freqs = (freqs >= 200) & (freqs <= 2000)
        melody_mask[melody_freqs] *= 1.5
        
        # Reduce very low and very high frequencies
        melody_mask[freqs < 100] *= 0.3
        melody_mask[freqs > 4000] *= 0.5
        
        # Apply mask
        magnitude_filtered = magnitude * melody_mask
        
        # Reconstruct audio
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        y_melody = librosa.istft(stft_filtered)
        
        return y_melody
    
    def _visualize_separation(self, original, harmonic, percussive, sr):
        """Visualize the harmonic-percussive separation."""
        plt.figure(figsize=(15, 10))
        
        # Time axis
        time = np.linspace(0, len(original) / sr, len(original))
        
        plt.subplot(3, 1, 1)
        plt.plot(time, original)
        plt.title("Original Audio")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time, harmonic)
        plt.title("Harmonic Component (Melody)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time, percussive)
        plt.title("Percussive Component (Drums/Rhythm)")
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("audio_preprocessing/melody_separation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_separation_with_vocals(self, original, harmonic, percussive, instrumental, sr):
        """Visualize the separation including vocal removal."""
        plt.figure(figsize=(15, 12))
        
        # Time axis
        time = np.linspace(0, len(original) / sr, len(original))
        
        plt.subplot(4, 1, 1)
        plt.plot(time, original)
        plt.title("Original Audio")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(time, harmonic)
        plt.title("Harmonic Component (Melody + Vocals)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(time, percussive)
        plt.title("Percussive Component (Drums/Rhythm)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(time, instrumental)
        plt.title("Instrumental Melody (Vocals Removed)")
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("audio_preprocessing/melody_separation_with_vocals.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def convert_to_midi(self, audio_file_path, output_midi_path=None):
        """
        Convert audio file to MIDI using Basic Pitch.
        
        Args:
            audio_file_path (str): Path to audio file (melody or full audio)
            output_midi_path (str): Path for output MIDI file
            
        Returns:
            str: Path to generated MIDI file
        """
        if output_midi_path is None:
            # Generate output path based on input file
            input_path = Path(audio_file_path)
            output_midi_path = input_path.parent / f"{input_path.stem}_melody.mid"
        
        print(f"Converting audio to MIDI using Basic Pitch...")
        print(f"Input: {audio_file_path}")
        print(f"Output: {output_midi_path}")
        
        # Use Basic Pitch to convert audio to MIDI
        model_output, midi_data, note_events = predict(
            audio_file_path,
            self.model_path
        )
        
        # Save MIDI file with custom name
        if midi_data is not None:
            midi_data.write(str(output_midi_path))
            print(f"MIDI file saved: {output_midi_path}")
        else:
            print("Warning: No MIDI data generated")
            
        return str(output_midi_path)
    
    def process_audio_to_midi(self, audio_file_path, output_dir="output", extract_melody=True):
        """
        Complete pipeline: Audio -> Melody Extraction -> MIDI
        
        Args:
            audio_file_path (str): Path to input audio file
            output_dir (str): Directory for output files
            extract_melody (bool): Whether to extract melody first or use full audio
            
        Returns:
            dict: Paths to generated files
        """
        print("=" * 60)
        print("AUDIO TO MIDI CONVERSION PIPELINE")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'input_audio': audio_file_path,
            'output_dir': output_dir
        }
        
        if extract_melody:
            print("\nStep 1: Extracting melody from audio (with vocal removal)...")
            melody_path, sr = self.extract_melody(audio_file_path, output_dir, remove_vocals=True)
            results['melody_audio'] = melody_path
            audio_for_midi = melody_path
        else:
            print("\nStep 1: Using full audio for MIDI conversion...")
            audio_for_midi = audio_file_path
        
        print("\nStep 2: Converting to MIDI...")
        midi_path = self.convert_to_midi(
            audio_for_midi, 
            os.path.join(output_dir, "output_melody.mid")
        )
        results['midi_file'] = midi_path
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"Input audio: {audio_file_path}")
        if extract_melody:
            print(f"Extracted melody: {results['melody_audio']}")
        print(f"MIDI output: {results['midi_file']}")
        
        return results

def main():
    """Example usage of the AudioToMIDIConverter."""
    converter = AudioToMIDIConverter()
    
    # Example with your sample file
    audio_file = "audio_preprocessing/mp3-files/sample-test-1.mp3"
    
    if os.path.exists(audio_file):
        print(f"Processing: {audio_file}")
        
        # Process with melody extraction
        results = converter.process_audio_to_midi(
            audio_file_path=audio_file,
            output_dir="audio_preprocessing/midi_output",
            extract_melody=True
        )
        
        print("\nGenerated files:")
        for key, path in results.items():
            if key != 'output_dir':
                print(f"  {key}: {path}")
    else:
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")

if __name__ == "__main__":
    main()
