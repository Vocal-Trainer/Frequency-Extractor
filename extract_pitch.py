import librosa
import numpy as np
import json

audio_path = "audio/Love_Me_Like_You_Do.wav"

# Load audio file
y, sr = librosa.load(audio_path)

# Get audio duration
duration_sec = librosa.get_duration(y=y, sr=sr)

# Estimate pitch using YIN algorithm
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Get times for pitch values
times = librosa.times_like(f0)

# Initialize empty list to store pitch data
pitch_data = []

# Loop through pitch and time values and append to pitch data list
for i in range(len(times)):
    time_sec = times[i]
    frequency_hz = np.nan_to_num(f0[i])
    pitch_data.append({"time": time_sec, "frequency":  frequency_hz })

# Create dictionary with duration and pitch data
output_dict = {"duration": duration_sec, "pitch": pitch_data}

# Convert dictionary to JSON format
output_json = json.dumps(output_dict)

# Print output to console
print(output_json)





