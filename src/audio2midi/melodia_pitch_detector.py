import numpy as np
from scipy.signal import medfilt
from pretty_midi_fix import PrettyMIDI,Instrument,Note
from librosa import load as librosa_load
from vamp import collect as vamp_collect

class Melodia():
    def midi_to_notes(self,midi, fs, smooth, minduration,hop=128):
        if (smooth > 0):
            filter_duration = smooth
            filter_size = int(filter_duration * fs / float(hop))
            if filter_size % 2 == 0:
                filter_size += 1
            midi_filt = medfilt(midi, filter_size)
        else:
            midi_filt = midi
        notes = []
        p_prev = 0
        duration = 0
        onset = 0
        for n, p in enumerate(midi_filt):
            if p == p_prev:
                duration += 1
            else:
                if p_prev > 0:
                    duration_sec = duration * hop / float(fs)
                    if duration_sec >= minduration:
                        onset_sec = onset * hop / float(fs)
                        notes.append((onset_sec, duration_sec, p_prev))
                onset = n
                duration = 1
                p_prev = p
        if p_prev > 0:
            duration_sec = duration * hop / float(fs)
            onset_sec = onset * hop / float(fs)
            notes.append((onset_sec, duration_sec, p_prev))
        return notes

    def hz2midi(self,hz:np.ndarray):
        hz_nonneg = hz.copy()
        idx = hz_nonneg <= 0
        hz_nonneg[idx] = 1
        midi = 69 + 12*np.log2(hz_nonneg/440.)
        midi[idx] = 0
        midi = np.round(midi)
        return midi

    def predict(self,audio, tempo=120, smooth=0.25, minduration=0.1,hop=128,output_file="output.mid"):
        data, sr = librosa_load(audio, sr=44100, mono=True)
        pm = PrettyMIDI(initial_tempo=tempo)
        instrument = Instrument(program=40)
        for onset_sec, duration_sec, pitch in self.midi_to_notes(
                self.hz2midi(np.insert(vamp_collect(data, sr, "mtg-melodia:melodia", parameters={"voicing": 0.2})['vector'][1],0, [0]*8)), 44100, smooth, minduration, hop):
            start = onset_sec
            end = start + duration_sec
            instrument.notes.append(Note(100, int(pitch), start, end))
        pm.instruments.append(instrument)
        pm.write(output_file)
        return output_file
