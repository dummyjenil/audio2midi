import librosa
from pretty_midi_fix import Instrument , PrettyMIDI , Note
import numpy as np


class Normal_Pitch_Det:
    def smooth_pitch_sequence(self, pitches, magnitudes, threshold):
        midi_sequence = []
        for i in range(pitches.shape[1]):
            index = np.argmax(magnitudes[:, i])
            pitch_mag = magnitudes[index, i]
            pitch = pitches[index, i]
            if pitch_mag < threshold or np.isnan(pitch) or pitch <= 0:
                midi_sequence.append(None)
            else:
                midi_note = int(round(librosa.hz_to_midi(pitch)))
                midi_sequence.append(midi_note)
        return midi_sequence

    def clean_midi_sequence(self, sequence, min_note_length):
        cleaned = []
        current_note = None
        count = 0
        for note in sequence + [None]:
            if note == current_note:
                count += 1
            else:
                if current_note is not None and count >= min_note_length:
                    cleaned.extend([current_note] * count)
                else:
                    cleaned.extend([None] * count)
                current_note = note
                count = 1
        return cleaned

    def predict(self, input_file, tempo_bpm=120, hop_length=512,min_note_length=2,threshold=0.1,output_file="output.mid"):
        wav, sr = librosa.load(input_file)
        audio_duration = len(wav) / sr
        pitches, magnitudes = librosa.piptrack(y=wav, sr=sr, hop_length=hop_length)
        midi_sequence = self.clean_midi_sequence(self.smooth_pitch_sequence(pitches, magnitudes,threshold),min_note_length)
        time_per_frame = audio_duration / len(midi_sequence)
        pm = PrettyMIDI(initial_tempo=tempo_bpm)
        instrument = Instrument(program=0)  # Acoustic Grand Piano
        last_note = None
        start_time = 0
        for i, note in enumerate(midi_sequence):
            current_time = i * time_per_frame
            if note != last_note:
                if last_note is not None:
                    end_time = current_time
                    instrument.notes.append(Note(velocity=100,pitch=last_note,start=start_time,end=end_time))
                if note is not None:
                    start_time = current_time
                last_note = note

        if last_note is not None:
            end_time = len(midi_sequence) * time_per_frame
            instrument.notes.append(Note(velocity=100,pitch=last_note,start=start_time,end=end_time))

        pm.instruments.append(instrument)
        pm.write(output_file)
        return output_file


class Guitar_Pitch_Det:
    def __init__(self):
        # nfft=2048
        # overlap=0.5
        # self.HOP_LENGTH = int(nfft * (1 - overlap))
        self.FMIN = librosa.note_to_hz('C1')

    def calc_cqt(self,audio, sr, mag_exp):
        """Compute CQT and convert to dB."""
        return librosa.amplitude_to_db(np.abs(librosa.cqt(audio, sr=sr, hop_length=self.HOP_LENGTH, fmin=self.FMIN, n_bins=self.N_BINS, bins_per_octave=self.BINS_PER_OCTAVE)) ** mag_exp, ref=np.max)

    def cqt_thresholded(self,cqt_db, threshold_db):
        """Threshold CQT in dB."""
        cqt_copy = np.copy(cqt_db)
        cqt_copy[cqt_copy < threshold_db] = -120
        return cqt_copy

    def calc_onset(self,cqt_db, sr, pre_post_max, backtrack):
        """Detect onsets using the onset envelope from thresholded CQT."""
        onset_env = librosa.onset.onset_strength(S=cqt_db, sr=sr, hop_length=self.HOP_LENGTH, aggregate=np.mean)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=self.HOP_LENGTH,units='frames', backtrack=backtrack,pre_max=pre_post_max, post_max=pre_post_max)
        boundaries = np.concatenate([[0], onset_frames, [cqt_db.shape[1]]])
        return boundaries, onset_env

    def estimate_segment_note(self,cqt_db, boundaries, i, sr, tempo_bpm, threshold_db,round_to_sixteenth):
        """Estimate pitch for one onset segment and generate note."""
        n0, n1 = int(boundaries[i]), int(boundaries[i + 1])
        # Pass the CQT segment directly to estimate_pitch
        segment_cqt = np.mean(cqt_db[:, n0:n1], axis=1)
        f0_info = self.estimate_pitch(segment_cqt, sr, threshold_db)
        return self.generate_note(cqt_db, tempo_bpm, f0_info, sr, n1 - n0,round_to_sixteenth)

    def generate_note(self,cqt_db, tempo_bpm, f0_info, sr, n_duration,round_to_sixteenth):
        """Generate sinewave, MIDI note data, and Note or Rest."""
        f0, amplitude = f0_info
        # Remap amplitude based on the range of cqt_db values for velocity mapping
        duration_beats = librosa.frames_to_time(n_duration, sr=sr, hop_length=self.HOP_LENGTH) * (tempo_bpm / 60)
        if round_to_sixteenth:
            duration_beats = round(duration_beats * 16) / 16
        # Remap amplitude based on the range of cqt_db values for MIDI velocity
        return None if f0 is None else int(np.round(librosa.hz_to_midi(f0))), duration_beats, int(np.clip(self.remap(amplitude, cqt_db.min(), cqt_db.max(), 0, 127), 0, 127))

    def estimate_pitch(self,segment_cqt, sr, threshold_db):
        """Estimate pitch from CQT segment."""
        # Analyze the CQT segment to find the dominant frequency
        # Find the frequency bin with the maximum energy in the segment
        max_bin = np.argmax(segment_cqt)
        # Convert the bin index to frequency (Hz)
        pitch_hz = librosa.cqt_frequencies(n_bins=self.N_BINS, fmin=self.FMIN, bins_per_octave=self.BINS_PER_OCTAVE)[max_bin]
        amplitude = segment_cqt[max_bin] # Use the amplitude from the CQT bin

        if pitch_hz is not None and amplitude > threshold_db:
            return pitch_hz, amplitude
        else:
            return None, 0

    def remap(self,x, in_min, in_max, out_min, out_max):
        """Remap a value or array from one range to another."""
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def predict(self,audio_path, mag_exp=4,threshold_db=-61,pre_post_max=6,backtrack=False,round_to_sixteenth=False,hop_length=1024,n_bins=72,bins_per_octave=12,output_file="output.mid"):
        self.BINS_PER_OCTAVE = bins_per_octave
        self.HOP_LENGTH = hop_length
        self.N_BINS = n_bins
        audio , sr = librosa.load(audio_path, sr=None)
        cqt_db = self.calc_cqt(audio, sr, mag_exp)
        cqt_thresh = self.cqt_thresholded(cqt_db, threshold_db)
        boundaries, onset_env = self.calc_onset(cqt_thresh, sr, pre_post_max, backtrack)
        tempo_bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,hop_length=self.HOP_LENGTH)
        tempo_bpm = round(tempo_bpm[0])
        # Process all segments
        notes_data = [self.estimate_segment_note(cqt_db, boundaries, i, sr, tempo_bpm, threshold_db,round_to_sixteenth) for i in range(len(boundaries) - 1)]
        pm = PrettyMIDI(initial_tempo=tempo_bpm)
        piano = Instrument(program=40)
        note_time = 0.0
        for (pitch, duration, velocity) in notes_data:
            if pitch is not None:
                # Convert duration in beats to duration in seconds for PrettyMIDI
                duration_sec = duration * (60 / tempo_bpm)
                piano.notes.append(Note(velocity, pitch, note_time, note_time + duration_sec))
                note_time += duration_sec # Increment note_time by duration in seconds
            else:
                # If it's a rest, just advance the time
                duration_sec = duration * (60 / tempo_bpm)
                note_time += duration_sec
        pm.instruments.append(piano)
        pm.write(output_file)
        return output_file
