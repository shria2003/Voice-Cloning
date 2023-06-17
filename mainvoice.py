import torch
import torchaudio
from scipy.io.wavfile import write

def clone_voice(text, output_path):
    # Load the Tacotron 2 and WaveGlow models
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')

    # Set the models to evaluation mode
    tacotron2.eval()
    waveglow.eval()

    # Load the pre-trained checkpoints for the models
    tacotron2_checkpoint = torch.load('tacotron2_checkpoint.pt')
    waveglow_checkpoint = torch.load('waveglow_checkpoint.pt')

    # Load the models with the checkpoints
    tacotron2.load_state_dict(tacotron2_checkpoint['state_dict'])
    waveglow.load_state_dict(waveglow_checkpoint['state_dict'])

    # Convert the text to phonemes (if required by the model)
    phonemes = tacotron2.text_to_sequence(text)

    # Convert the phonemes to a tensor
    phoneme_tensor = torch.from_numpy(phonemes).unsqueeze(0)

    # Generate mel spectrogram from the phoneme tensor using Tacotron 2
    mel_outputs, mel_lengths, _, _ = tacotron2.infer(phoneme_tensor)

    # Generate audio from the mel spectrogram using WaveGlow
    audio = waveglow.infer(mel_outputs)

    # Convert the audio tensor to a numpy array
    audio_np = audio[0].data.cpu().numpy()

    # Save the audio as a WAV file
    write(output_path, 22050, audio_np)

# Example usage
text_to_synthesize = "Hello, how are you?"
output_file_path = "cloned_voice.wav"

clone_voice(text_to_synthesize, output_file_path)
print("Voice cloning complete. Cloned voice saved as 'cloned_voice.wav'.")
