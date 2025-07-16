from typing import Callable
import numpy as np
import pretty_midi_fix
import tensorflow
import librosa.feature.rhythm
from huggingface_hub import hf_hub_download
import librosa


def endpitch(pitch, endFrame,volProb,intervals,MIN_NOTE_FRAMES,frameLenSecs,PITCH_BEND_ENABLED,pitch_bend_steps,pitch_bend_depth,track):
    startFrame = intervals[pitch]
    if endFrame - startFrame < MIN_NOTE_FRAMES:
        return
    vol = volProb[startFrame, pitch]
    if vol < 0 or vol > 1:
        return
    start_time = startFrame * frameLenSecs
    track.notes.append(pretty_midi_fix.Note(velocity=int(max(0, min(1, vol)) * 80 + 10), pitch=pitch + 21, start=start_time, end=endFrame * frameLenSecs))
    if PITCH_BEND_ENABLED:
        for step in range(pitch_bend_steps):
            track.pitch_bends.append(pretty_midi_fix.PitchBend(pitch=int(np.sin(np.pi * step / (pitch_bend_steps - 1)) * pitch_bend_depth), time=start_time + step * 0.01))
        track.pitch_bends.append(pretty_midi_fix.PitchBend(pitch=0, time=start_time + 0.05))  # Reset
    del intervals[pitch]

def model_output_to_notes(model_output,onset_thresh,include_pitch_bends,min_note_len,gap_tolerance_frames,pitch_bend_depth,pitch_bend_steps):
    actProb , onProb , offProb , volProb , tempo = model_output
    midi = pretty_midi_fix.PrettyMIDI(initial_tempo=tempo)
    track = pretty_midi_fix.Instrument(program=40)
    frameLenSecs = librosa.frames_to_time(1, sr=16000)
    intervals = {}
    onsets = (onProb > onset_thresh).astype(np.int8)
    frames = onsets | (actProb > onset_thresh).astype(np.int8)
    for i, frame in enumerate(np.vstack([frames, np.zeros(frames.shape[1])])):
        for pitch, active in enumerate(frame):
            if active:
                if pitch not in intervals:
                    if onsets is None or onsets[i, pitch]:
                        intervals[pitch] = i
                elif onsets is not None and onsets[i, pitch] and (i - intervals[pitch] > 2):
                    endpitch(pitch, i,volProb,intervals,min_note_len,frameLenSecs,include_pitch_bends,pitch_bend_steps,pitch_bend_depth,track)
                    intervals[pitch] = i
            elif pitch in intervals:
                if i + gap_tolerance_frames < frames.shape[0] and np.any(frames[i:i + gap_tolerance_frames, pitch]):
                    continue  # Don't end the note yet
                endpitch(pitch, i,volProb,intervals,min_note_len,frameLenSecs,include_pitch_bends,pitch_bend_steps,pitch_bend_depth,track)
    midi.instruments.append(track)
    return midi

class Magenta:
    def __init__(self,model_path=hf_hub_download("shethjenil/Audio2Midi_Models","magenta.tflite")):
        self.interp = tensorflow.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inputLen = self.interp.get_input_details()[0]['shape'][0]
        self.outputStep = self.interp.get_output_details()[0]['shape'][1] * 512

    def run_inference(self,audio_path,progress_callback):
        song = librosa.load(audio_path,sr=16000)[0]
        actProb, onProb, offProb, volProb = np.empty((1, 88)), np.empty((1, 88)), np.empty((1, 88)), np.empty((1, 88))
        paddedSong = np.append(song, np.zeros(-(song.size - self.inputLen) % self.outputStep, dtype=np.float32))
        total_size = (paddedSong.size - self.inputLen) // self.outputStep + 1
        tempo = librosa.feature.rhythm.tempo(y=song, sr=16000).mean()
        for i in range(total_size):
            self.interp.set_tensor(self.interp.get_input_details()[0]['index'], paddedSong[i * self.outputStep : i * self.outputStep + self.inputLen])
            self.interp.invoke()
            actProb = np.vstack((actProb, self.interp.get_tensor(self.interp.get_output_details()[0]['index'])[0]))
            onProb  = np.vstack(( onProb, self.interp.get_tensor(self.interp.get_output_details()[1]['index'])[0]))
            offProb = np.vstack((offProb, self.interp.get_tensor(self.interp.get_output_details()[2]['index'])[0]))
            volProb = np.vstack((volProb, self.interp.get_tensor(self.interp.get_output_details()[3]['index'])[0]))
            if progress_callback:
                progress_callback(i,total_size)
        return actProb , onProb , offProb , volProb , tempo


    def predict(self,audio,onset_thresh=0,min_note_len=3,gap_tolerance_frames = 3,pitch_bend_depth = 1500,pitch_bend_steps = 4,include_pitch_bends=True,progress_callback: Callable[[int, int], None] = None,output_file="output.mid"):
        model_output_to_notes(self.run_inference(audio,progress_callback),onset_thresh  = onset_thresh,min_note_len  = min_note_len,include_pitch_bends  = include_pitch_bends,pitch_bend_depth=pitch_bend_depth,pitch_bend_steps=pitch_bend_steps,gap_tolerance_frames=gap_tolerance_frames).write(output_file)
        return output_file
