import numpy as np
import tensorflow as tf
import librosa
from scipy.signal import find_peaks
from huggingface_hub import snapshot_download
import pretty_midi_fix
from tqdm import tqdm
from collections import deque

# --- Config ---
EXPECTED_WIDTH = 626
EXPECTED_HEIGHT = 229
HOP_LENGTH = 512
SR = 16000

# Load model once
model = tf.saved_model.load(snapshot_download("Razvanix/music-transcription-model")).signatures["serving_default"]

def preprocess_mel(chunk_audio, sr=SR):
    """Convert audio chunk to fixed-size mel spectrogram tensor for the model."""
    mel = librosa.feature.melspectrogram(
        y=chunk_audio, sr=sr, n_mels=EXPECTED_HEIGHT,
        hop_length=HOP_LENGTH, n_fft=2048
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize/pad to expected size
    if mel_db.shape[0] != EXPECTED_HEIGHT:
        mel_db = tf.image.resize(tf.expand_dims(mel_db, 0), [EXPECTED_HEIGHT, mel_db.shape[1]])[0]
    if mel_db.shape[1] < EXPECTED_WIDTH:
        mel_db = np.pad(mel_db, ((0, 0), (0, EXPECTED_WIDTH - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :EXPECTED_WIDTH]

    # Model expects (1, time, freq, 1)
    mel_input = tf.transpose(mel_db)
    mel_input = tf.expand_dims(mel_input, axis=0)
    mel_input = tf.expand_dims(mel_input, axis=-1)
    return mel_input


def transcribe_audio_to_midi(
    audio_path, model,
    onset_threshold=0.3, frame_threshold=0.3, offset_threshold=0.3,
    min_note_duration=0.05, max_note_duration=8.0,
    time_resolution=0.032,allow_offset_dense=False,
):
    """Transcribe audio file into MIDI using a pretrained transcription model.
    
    Uses offset-based duration if available, otherwise falls back to frame-based logic.
    """
    y, sr = librosa.load(audio_path, sr=SR)
    total_duration = len(y) / sr
    chunk_duration_sec = (EXPECTED_WIDTH * HOP_LENGTH) / sr
    overlap_duration_sec = 2.0

    midi = pretty_midi_fix.PrettyMIDI()
    instrument = pretty_midi_fix.Instrument(program=0)

    recent_notes = deque(maxlen=10)
    n_chunks = int(np.ceil(total_duration / (chunk_duration_sec - overlap_duration_sec)))

    start_time = 0.0
    for _ in tqdm(range(n_chunks), desc="Transcribing"):
        end_time = min(start_time + chunk_duration_sec, total_duration)
        start_sample, end_sample = int(start_time * sr), int(end_time * sr)
        chunk_audio = y[start_sample:end_sample]

        mel_input = preprocess_mel(chunk_audio, sr)
        predictions = model(mel_input)

        onset = predictions.get("onset_dense", [None])[0]
        frame = predictions.get("frame_dense", [None])[0]
        velocity = predictions.get("velocity_dense", [None])[0]
        offset = predictions.get("offset_dense", [None])[0] if allow_offset_dense else None

        for pitch in range(min(88, onset.shape[1])):
            # Find onset peaks
            peaks, _ = find_peaks(
                onset[:, pitch],
                height=onset_threshold,
                distance=max(1, int(0.05 / time_resolution))
            )
            for peak in peaks:
                # --- Duration Logic ---
                if offset is not None:  
                    # Use offset-based detection
                    offsets, _ = find_peaks(
                        offset[peak:, pitch],
                        height=offset_threshold,
                        distance=max(1, int(0.05 / time_resolution))
                    )
                    if len(offsets) > 0:
                        dur = offsets[0] * time_resolution
                    else:
                        # Fallback to frame if no offset peak found
                        frames = frame[peak:, pitch]
                        dur_frames = np.where(frames < frame_threshold)[0]
                        dur = (dur_frames[0] if len(dur_frames) else int(0.5 / time_resolution)) * time_resolution
                else:
                    # Pure frame-based duration
                    frames = frame[peak:, pitch]
                    dur_frames = np.where(frames < frame_threshold)[0]
                    dur = (dur_frames[0] if len(dur_frames) else int(0.5 / time_resolution)) * time_resolution

                dur = np.clip(dur, min_note_duration, max_note_duration)
                if dur < min_note_duration:
                    continue

                vel = float(np.clip(velocity[peak, pitch], 0, 1)) if velocity is not None else 0.8
                midi_pitch = pitch + 21
                note_start = round(peak * time_resolution + start_time, 6)
                note_end = note_start + round(dur, 6)

                # Avoid duplicate notes
                if any(midi_pitch == r['pitch'] and abs(note_start - r['start']) < 0.5 for r in recent_notes):
                    continue

                instrument.notes.append(pretty_midi_fix.Note(
                    velocity=int(min(127, max(1, vel * 127))),
                    pitch=midi_pitch,
                    start=note_start,
                    end=note_end
                ))
                recent_notes.append({'pitch': midi_pitch, 'start': note_start})

        if end_time >= total_duration:
            break
        start_time = end_time - overlap_duration_sec

    midi.instruments.append(instrument)
    return midi




transcribe_audio_to_midi("path/to/audio.wav", model).write("output.mid")






def clean_up_notes(notes_list, min_duration=0.05, merge_gap=0.08, confidence_threshold=0.4):
    """Filter and merge notes to improve MIDI quality - ORIGINAL"""
    filtered_notes = []
    for note in notes_list:
        if note["duration"] >= min_duration and note["velocity"] >= confidence_threshold:
            filtered_notes.append(note)

    filtered_notes.sort(key=lambda x: (x["pitch"], x["time"]))

    merged_notes = []
    i = 0
    while i < len(filtered_notes):
        current_note = filtered_notes[i]

        j = i + 1
        while j < len(filtered_notes) and filtered_notes[j]["pitch"] == current_note["pitch"]:
            next_note = filtered_notes[j]

            gap = next_note["time"] - (current_note["time"] + current_note["duration"])
            if gap <= merge_gap:
                current_note["duration"] = (next_note["time"] + next_note["duration"]) - current_note["time"]
                current_note["velocity"] = max(current_note["velocity"], next_note["velocity"])
                current_note["velocity_midi"] = max(current_note["velocity_midi"], next_note["velocity_midi"])
                j += 1
            else:
                break

        merged_notes.append(current_note)
        i = j

    return merged_notes
