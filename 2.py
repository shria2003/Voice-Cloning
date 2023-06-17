# 1) Cloning the voice with ascent for example Indian Hindi and American English 
# 2) Changing the feature of the voice to make it a more pleasing sound for example compare with the best voices in the world and add those features into it. 
# 3) Add filters like authority, humbleness, etc. 
# 4) Users can use it for singing - use can do karaoke and get pleasing songs - for example, I sing a song and the model will give me a better sound with music.

import torchaudio
import torchaudio.transforms as T

# Load the audio file
waveform, sample_rate = torchaudio.load('original_voice.wav')

# Apply pitch shifting transformation
transform = T.PitchShift(n_steps=2)
transformed_waveform = transform(waveform)

# Save the transformed audio
torchaudio.save('pleasing_voice.wav', transformed_waveform, sample_rate)
