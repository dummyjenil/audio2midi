from keras.layers import Input, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
from keras.models import Model
from keras.callbacks import Callback
from hmmlearn.hmm import CategoricalHMM
from math import ceil as math_ceil
from typing import Callable
from numpy.lib.stride_tricks import as_strided
from librosa import load as librosa_load
from pretty_midi_fix import PrettyMIDI , PitchBend , Note ,Instrument
import numpy as np
from huggingface_hub import hf_hub_download

class PredictProgressCallback(Callback):
    def __init__(self, total_batches,progress_callback: Callable[[int, int], None] = None):
        super().__init__()
        self.total_batches = total_batches
        self.progress_callback = progress_callback
    def on_predict_begin(self, logs=None):
        if self.progress_callback:
            self.progress_callback(0,self.total_batches)
    def on_predict_batch_end(self, batch, logs=None):
        if self.progress_callback:
            self.progress_callback(batch,self.total_batches)
    def on_predict_end(self, logs=None):
        if self.progress_callback:
            self.progress_callback(self.total_batches,self.total_batches)


class CrepeTF():
    def __init__(self,model_type="full",model_path=None):
        if not model_path:
            model_path = hf_hub_download("shethjenil/Audio2Midi_Models",f"crepe_{model_type}.h5")
        model_type_importance = {'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32}[model_type]
        filters = [n * model_type_importance for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        x = Input(shape=(1024,), name='input', dtype='float32')
        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)
        layers = [1, 2, 3, 4, 5, 6]
        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same', activation='relu', name="conv%d" % l)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid', name="conv%d-maxpool" % l)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)
        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation='sigmoid', name="classifier")(y)
        self.model = Model(inputs=x, outputs=y)
        self.model.load_weights(model_path)
        self.model.compile('adam', 'binary_crossentropy')
        self.cents_mapping=(np.linspace(0, 7180, 360) + 1997.3794084376191)

    def to_local_average_cents(self, salience, center=None):
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.cents_mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            return np.array([self.to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")

    def to_viterbi_cents(self,salience):
        starting = np.ones(360) / 360
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / np.sum(transition, axis=1)[:, None]
        self_emission = 0.1
        emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 360))
        model = CategoricalHMM(360, starting, transition)
        model.startprob_, model.transmat_, model.emissionprob_ = starting, transition, emission
        observations = np.argmax(salience, axis=1)
        path = model.predict(observations.reshape(-1, 1), [len(observations)])
        return np.array([self.to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])

    def get_activation(self,audio:np.ndarray,center, step_size, progress_callback,batch_size):
        if center:
            audio = np.pad(audio, 512, mode='constant', constant_values=0)
        hop_length = int(16000 * step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(audio, shape=(1024, n_frames),strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.transpose().copy()
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
        return self.model.predict(frames,batch_size,0,callbacks=[PredictProgressCallback(math_ceil(len(frames) / batch_size),progress_callback)])
    
    def model_predict(self,audio:np.ndarray,viterbi, center, step_size,progress_callback,batch_size):
        activation = self.get_activation(audio.astype(np.float32), center, step_size,progress_callback,batch_size)
        confidence = activation.max(axis=1)
        cents = self.to_viterbi_cents(activation) if viterbi else self.to_local_average_cents(activation)
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0
        time = np.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence

    def predict(self,audio_path,viterbi=False, center=True, step_size=10,min_confidence=0.8,batch_size=32,progress_callback: Callable[[int, int], None] = None,output_file= "output.mid"):
        time, frequency, confidence = self.model_predict(librosa_load(audio_path, sr=16000, mono=True)[0],viterbi,center,step_size,progress_callback,batch_size)
        mask = confidence > min_confidence
        times = time[mask]
        frequencies = frequency[mask]
        midi_floats = 69 + 12 * np.log2(frequencies / 440.0)
        midi_notes = np.round(midi_floats).astype(int)
        pitch_offsets = midi_floats - midi_notes  # in semitones
        midi = PrettyMIDI()
        instrument = Instrument(program=40)  # e.g., Violin for pitch bend demo
        if len(times) > 0:
            current_note = midi_notes[0]
            note_start = times[0]
            for i in range(1, len(times)):
                if midi_notes[i] != current_note or i == len(times) - 1:
                    note_end = times[i]
                    if 0 <= current_note <= 127:
                        note = Note(velocity=100,pitch=int(current_note),start=note_start,end=note_end)
                        instrument.notes.append(note)
                        seg_mask = (times >= note_start) & (times <= note_end)
                        seg_times = times[seg_mask]
                        seg_offsets = pitch_offsets[seg_mask]
                        for t, offset in zip(seg_times, seg_offsets):
                            # Assuming pitch bend range is +/- 2 semitones
                            bend_value = int(offset / 2.0 * 8192)  # Scale to -8192 to +8191
                            bend_value = np.clip(bend_value, -8192, 8191)
                            pb = PitchBend(pitch=bend_value, time=t)
                            instrument.pitch_bends.append(pb)
                    current_note = midi_notes[i]
                    note_start = times[i]
        midi.instruments.append(instrument)
        midi.write(output_file)
        return output_file
