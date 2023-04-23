import librosa
import numpy as np
import json
from typing import List, Dict

def group_by_second_and_get_median(pitch_data: Dict) -> List[Dict]:
    grouped = {}
    for pitch in pitch_data:
        time = int(pitch['time'])
        frequency = pitch['frequency']
        if time in grouped:
            grouped[time].append(frequency)
        else:
            grouped[time] = [frequency]

    result = []
    for time, frequencies in grouped.items():
        median_frequency = sum(frequencies) / len(frequencies)
        result.append({
            'time': time,
            'frequency': median_frequency
        })

    return result


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

# Get the median pitch for each second of the recording
pitch_median = group_by_second_and_get_median(pitch_data)

# Create dictionary with duration and pitch data
output_dict = {"duration": duration_sec, "pitch": pitch_median}

# Convert dictionary to JSON format
output_json = json.dumps(output_dict)

# Print output to console
print(output_json)



