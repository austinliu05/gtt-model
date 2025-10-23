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
        
    def extract_melody(self, audio_file_path, output_dir="temp_audio"):
        """
        Extract melody from audio file using harmonic-percussive separation.
        
        Args:
            audio_file_path (str): Path to input audio file
            output_dir (str): Directory to save processed audio
            
        Returns:
            str: Path to extracted melody audio file
        """
        print(f"Loading audio file: {audio_file_path}")
        y, sr = librosa.load(audio_file_path, sr=None)
        
        print("Separating harmonic (melody) from percussive components...")
        # Use HPSS to separate melody from drums/percussion
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the harmonic (melody) component
        melody_file_path = os.path.join(output_dir, "extracted_melody.wav")
        sf.write(melody_file_path, y_harmonic, sr)
        
        print(f"Melody extracted and saved to: {melody_file_path}")
        
        # Visualize the separation
        self._visualize_separation(y, y_harmonic, y_percussive, sr)
        
        return melody_file_path, sr
    
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
        plt.savefig("audio_processing/melody_separation.png", dpi=300, bbox_inches='tight')
        plt.show()
    
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
            self.model_path,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=True,
            output_directory=os.path.dirname(output_midi_path),
            save_notes=True
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
            print("\nStep 1: Extracting melody from audio...")
            melody_path, sr = self.extract_melody(audio_file_path, output_dir)
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
    audio_file = "audio-processing/mp3-files/sample-test-1.mp3"
    
    if os.path.exists(audio_file):
        print(f"Processing: {audio_file}")
        
        # Process with melody extraction
        results = converter.process_audio_to_midi(
            audio_file_path=audio_file,
            output_dir="audio-processing/midi_output",
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
