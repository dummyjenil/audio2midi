from typing import Callable
import numpy as np
import tensorflow as tf
import librosa
from scipy.signal import find_peaks
from huggingface_hub import snapshot_download
import pretty_midi_fix
from collections import deque

# ------------------------------------------------------------
# 🎵 Convert Predictions → MIDI Notes
# ------------------------------------------------------------
def output_to_midi(
    all_predictions,
    output_file="output.mid",
    time_resolution=0.032,
    onset_thresh=0.3,
    frame_thresh=0.3,
    offset_threshold=0.3,
    min_note_len=0.05,
    max_note_duration=8.0,
):
    """Convert model predictions to MIDI."""
    raw_notes = []
    recent_notes = deque(maxlen=10)
    overlap_duration_sec = 2.0
    chunk_duration_sec = (626 * 512) / 16000
    start_time = 0.0

    for preds in all_predictions:
        onset, frame, velocity, offset = (
            preds["onset"], preds["frame"], preds["velocity"], preds["offset"]
        )

        for pitch in range(min(88, onset.shape[1])):
            peaks, _ = find_peaks(onset[:, pitch], height=onset_thresh,
                                    distance=max(1, int(0.05 / time_resolution)))
            for peak in peaks:
                # Duration calculation
                if offset is not None:
                    offsets, _ = find_peaks(
                        offset[peak:, pitch],
                        height=offset_threshold,
                        distance=max(1, int(0.05 / time_resolution))
                    )
                    if len(offsets) > 0:
                        dur = offsets[0] * time_resolution
                    else:
                        frames = frame[peak:, pitch]
                        dur_frames = np.where(frames < frame_thresh)[0]
                        dur = (dur_frames[0] if len(dur_frames) else int(0.5 / time_resolution)) * time_resolution
                else:
                    frames = frame[peak:, pitch]
                    dur_frames = np.where(frames < frame_thresh)[0]
                    dur = (dur_frames[0] if len(dur_frames) else int(0.5 / time_resolution)) * time_resolution

                dur = np.clip(dur, min_note_len, max_note_duration)
                if dur < min_note_len:
                    continue

                vel = float(np.clip(velocity[peak, pitch], 0, 1)) if velocity is not None else 0.8
                midi_pitch = pitch + 21
                note_start = round(peak * time_resolution + start_time, 6)

                # Avoid duplicates
                if any(midi_pitch == r['pitch'] and abs(note_start - r['time']) < 0.5 for r in recent_notes):
                    continue

                raw_notes.append({
                    "pitch": midi_pitch,
                    "time": note_start,
                    "duration": dur,
                    "velocity": vel,
                    "velocity_midi": int(min(127, max(1, vel * 127)))
                })
                recent_notes.append({'pitch': midi_pitch, 'time': note_start})

        start_time += chunk_duration_sec - overlap_duration_sec

    # Build MIDI
    midi = pretty_midi_fix.PrettyMIDI()
    instrument = pretty_midi_fix.Instrument(program=0)
    for note in raw_notes:
        instrument.notes.append(pretty_midi_fix.Note(
            velocity=note["velocity_midi"],
            pitch=note["pitch"],
            start=note["time"],
            end=note["time"] + note["duration"]
        ))
    midi.instruments.append(instrument)
    midi.write(output_file)
    return output_file



class TF_Transcription:

    def __init__(self, model_path=None):
        if not model_path:
            model_path = snapshot_download("Razvanix/music-transcription-model")
        self.model = tf.saved_model.load(model_path).signatures["serving_default"]
        self.width = 626
        self.height = 229
        self.hop_len = 512

    # ------------------------------------------------------------
    # 🧩 Preprocessing
    # ------------------------------------------------------------
    def preprocess_mel(self, chunk_audio):
        mel = librosa.feature.melspectrogram(y=chunk_audio, sr=16000, n_mels=self.height)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[0] != self.height:
            mel_db = tf.image.resize(tf.expand_dims(mel_db, 0), [self.height, mel_db.shape[1]])[0]
        if mel_db.shape[1] < self.width:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.width - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :self.width]
        mel_input = tf.transpose(mel_db)
        mel_input = tf.expand_dims(mel_input, axis=0)
        mel_input = tf.expand_dims(mel_input, axis=-1)
        return mel_input

    # ------------------------------------------------------------
    # 🧠 Modified: Batched model prediction
    # ------------------------------------------------------------
    def predict_model_output(self, audio, progress_callback: Callable[[int, int], None] = None, batch_size=4):
        """
        Run model inference in batches on preprocessed audio chunks.
        Returns list of prediction dicts for each chunk.
        """
        mel_batches = self.preprocess_audio(audio)
        n_chunks = mel_batches.shape[0]
        predictions_all = []

        for i in range(0, n_chunks, batch_size):
            batch = mel_batches[i:i+batch_size]
            preds = self.model(batch)

            for b in range(batch.shape[0]):
                predictions_all.append({
                    "onset": preds["onset_dense"][b].numpy(),
                    "frame": preds["frame_dense"][b].numpy(),
                    "velocity": preds["velocity_dense"][b].numpy(),
                    "offset": preds["offset_dense"][b].numpy() if "offset_dense" in preds else None
                })
            if progress_callback:
                progress_callback(i + 1, n_chunks)

        return predictions_all

    # ------------------------------------------------------------
    # 🎚️ New: Split full audio into fixed-size chunks for batching
    # ------------------------------------------------------------
    def preprocess_audio(self, audio, overlap_duration=2.0):
        """
        Split full audio into overlapping chunks ready for model input.
        Returns list of mel spectrogram tensors (one per chunk).
        """
        chunk_duration = (self.width * self.hop_len) / 16000
        step = chunk_duration - overlap_duration
        total_duration = len(audio) / 16000

        chunks = []
        start_time = 0.0

        while start_time < total_duration:
            end_time = min(start_time + chunk_duration, total_duration)
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            chunk_audio = audio[start_sample:end_sample]
            mel_input = self.preprocess_mel(chunk_audio)
            chunks.append(mel_input)
            start_time += step

        return tf.concat(chunks, axis=0)  # shape: (n_chunks, width, height, 1)

    # ------------------------------------------------------------
    # 🚀 Final: Full pipeline using batched prediction
    # ------------------------------------------------------------
    def predict(self, audio_path,
        onset_thresh=0.1,
        frame_thresh=0.3,
        min_note_len=0.05,
        offset_threshold=0.3,
        max_note_duration=8.0,
        time_resolution=0.032,
        batch_size=4,
        progress_callback: Callable[[int, int], None] = None,
        output_file="output.mid",
        ):
        """
        Complete pipeline: audio → preprocessing → batched model prediction → MIDI.
        """
        return output_to_midi(
            self.predict_model_output(librosa.load(audio_path, sr=16000)[0], progress_callback=progress_callback,batch_size=batch_size),
        output_file=output_file,
        time_resolution=time_resolution,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        offset_threshold=offset_threshold,
        min_note_len=min_note_len,
        max_note_duration=max_note_duration,
        )