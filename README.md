# Audio2Midi

```bash
pip install audio2midi[all] audio2midi[pop2piano] audio2midi[violin_pitch_detector] audio2midi[crepe_pitch_detector] audio2midi[crepe_pitch_detector_tf] audio2midi[melodia_pitch_detector] audio2midi[basic_pitch_pitch_detector] audio2midi[librosa_pitch_detector] audio2midi[magenta_music_transcription] audio2midi[yourmt3_music_transcription] audio2midi[mt3_music_transcription]
```

[Audio2Midi Demo](https://huggingface.co/spaces/shethjenil/Audio2Midi)

[Github](https://github.com/dummyjenil/audio2midi)

---

crepe_model_capacity

* tiny
* small
* medium
* large
* full

---

``` python
from audio2midi.librosa_pitch_detector import Normal_Pitch_Det , Guitar_Pitch_Det

audio_path = "audio.mp3"
Normal_Pitch_Det().predict(audio_path)
Guitar_Pitch_Det().predict(audio_path)
```

---

``` python
from os import environ
from huggingface_hub import hf_hub_download
from shutil import unpack_archive
from pathlib import Path
from audio2midi.melodia_pitch_detector import Melodia
from platform import system as platform_system , architecture as platform_architecture

import nest_asyncio
from audio2midi.mt3_music_transcription import MT3
from yourmt3_music_transcription import YMT3
nest_asyncio.apply()
unpack_archive(hf_hub_download("shethjenil/Audio2Midi_Models","mt3.zip"),"mt3_model",format="zip")
MT3("mt3_model").predict(audio_path)
name = "YMT3+"
YMT3(hf_hub_download("shethjenil/Audio2Midi_Models",f"{name}.pt"),name,"32" if str(device) == "cpu" else "16").predict("audio.mp3")

unpack_archive(hf_hub_download("shethjenil/Audio2Midi_Models",f"melodia_vamp_plugin_{'win' if (system := platform_system()) == 'Windows' else 'mac' if system == 'Darwin' else 'linux64' if (arch := platform_architecture()[0]) == '64bit' else 'linux32' if arch == '32bit' else None}.zip"),"vamp_melodia",format="zip")
environ['VAMP_PATH'] = str(Path("vamp_melodia").absolute())
Melodia().predict(audio_path)
```

---

```python
from audio2midi.basic_pitch_pitch_detector import BasicPitch
from audio2midi.crepe_pitch_detector import Crepe
from audio2midi.violin_pitch_detector import Violin_Pitch_Det
from audio2midi.pop2piano import Pop2Piano
from audio2midi.magenta_music_transcription import Magenta
from torch import device as Device
from torch.cuda import is_available as cuda_is_available
device = Device("cuda" if cuda_is_available() else "cpu")
Crepe().predict(audio_path)
Pop2Piano(device=device).predict(audio_path)
Violin_Pitch_Det(device=device).predict(audio_path)
BasicPitch(device=device).predict(audio_path)
Magenta().predict(audio_path)
```

---

```python
from audio2midi.basic_pitch_pitch_detector import BasicPitch
from audio2midi.crepe_pitch_detector_tf import CrepeTF
from audio2midi.crepe_pitch_detector import Crepe
from audio2midi.librosa_pitch_detector import Normal_Pitch_Det , Guitar_Pitch_Det
from audio2midi.melodia_pitch_detector import Melodia
from audio2midi.pop2piano import Pop2Piano
from audio2midi.violin_pitch_detector import Violin_Pitch_Det
from audio2midi.yourmt3_music_transcription import YMT3
from audio2midi.magenta_music_transcription import Magenta
from os import environ
from huggingface_hub import hf_hub_download
from shutil import unpack_archive
from pathlib import Path
from platform import system as platform_system , architecture as platform_architecture

unpack_archive(hf_hub_download("shethjenil/Audio2Midi_Models",f"melodia_vamp_plugin_{'win' if (system := platform_system()) == 'Windows' else 'mac' if system == 'Darwin' else 'linux64' if (arch := platform_architecture()[0]) == '64bit' else 'linux32' if arch == '32bit' else None}.zip"),"vamp_melodia",format="zip")

environ['VAMP_PATH'] = str(Path("vamp_melodia").absolute())

from os import getenv
from torch import device as Device
from torch.cuda import is_available as cuda_is_available
device = Device("cuda" if cuda_is_available() else "cpu")

import gradio as gr
with gr.Blocks() as midi_viz_ui:
    midi = gr.File(label="Upload MIDI")
    sf = gr.File(label="Upload SoundFont")
    output_html = gr.HTML(f'''
    <div style="display: flex; justify-content: center; align-items: center;">
        <iframe style="width: 100%; height: 500px;" src="https://shethjenil-midivizsf2.static.hf.space/index_single_file.html" id="midiviz"></iframe>
    </div>''')
    midi.upload(None, inputs=midi, js="""
    async (file) => {
        if (!file || !file.url || !file.orig_name) return;
        const iframe = document.getElementById("midiviz");
        iframe.contentWindow.postMessage({
            type: "load-midi",
            url: file.url,
            name: file.orig_name
        }, "*");
    }
    """)
    sf.upload(None, inputs=sf, js="""
    async (file) => {
        if (!file || !file.url || !file.orig_name) return;
        const iframe = document.getElementById("midiviz");
        iframe.contentWindow.postMessage({
            type: "load-sf",
            url: file.url,
            name: file.orig_name
        }, "*");
    }
    """)

gr.TabbedInterface([
    gr.Interface(Normal_Pitch_Det().predict,[gr.Audio(type="filepath",label="Input Audio"),gr.Number(120,label="BPM"),gr.Number(512,label="HOP Len"),gr.Number(2,label="minimum note length"),gr.Number(0.1,label="threshold")],gr.File(label="Midi File")),
    gr.Interface(Guitar_Pitch_Det().predict,[gr.Audio(type="filepath",label="Input Audio"),gr.Number(4,label="mag_exp"),gr.Number(-61,label="Threshold"),gr.Number(6,label="Pre_post_max"),gr.Checkbox(False,label="backtrack"),gr.Checkbox(False,label="round_to_sixteenth"),gr.Number(1024,label="hop_length"),gr.Number(72,label="n_bins"),gr.Number(12,label="bins_per_octave")],gr.File(label="Midi File")),
    gr.Interface(Melodia().predict,[gr.Audio(type="filepath",label="Input Audio"),gr.Number(120,label="BPM",step=30),gr.Number(0.25,label="smoothness",step=0.05,info="Smooth the pitch sequence with a median filter of the provided duration (in seconds)."),gr.Number(0.1,label="minimum duration",step=0.1,info="Minimum allowed duration for note (in seconds). Shorter notes will be removed."),gr.Number(128,label="HOP")],gr.File(label="Midi File")),
    gr.Interface(BasicPitch(device).predict,[gr.Audio(type="filepath", label="Upload Audio"),gr.Number(0.5,label="onset_thresh",info="Minimum amplitude of an onset activation to be considered an onset."),gr.Number(0.3,label="frame_thresh",info="Minimum energy requirement for a frame to be considered present."),gr.Number(11,label="min_note_len",info="The minimum allowed note length in milliseconds."),gr.Number(120,label="midi_tempo"),gr.Checkbox(True,label="infer_onsets",info="add additional onsets when there are large differences in frame amplitudes."),gr.Checkbox(True,label="include_pitch_bends",info="include pitch bends."),gr.Checkbox(False,label="multiple_pitch_bends",info="allow overlapping notes in midi file to have pitch bends."),gr.Checkbox(True,label="melodia_trick",info="Use the melodia post-processing step.")],gr.File(label="Download Midi File")),
    gr.Interface(Magenta().predict,[gr.Audio(type="filepath", label="Upload Audio"),gr.Number(0.5,label="onset_thresh",info="Minimum amplitude of an onset activation to be considered an onset."),gr.Number(3,label="min_note_len",info="The minimum allowed note length"),gr.Number(3,label="gap_tolerance_frames"),gr.Number(4,label="pitch_bend_steps"),gr.Number(1500,label="pitch_bend_depth"),gr.Checkbox(True,label="include_pitch_bends",info="include pitch bends.")],gr.File(label="Download Midi File")),
    gr.Interface(Violin_Pitch_Det(device).predict, [gr.Audio(label="Upload your Audio file",type="filepath"),gr.Number(32,label="Batch size"),gr.Radio(["spotify","tiktok"],value="spotify",label="Post Processing"),gr.Checkbox(True,label="include_pitch_bends")],gr.File(label="Download MIDI file")),
    gr.Interface(Crepe(getenv("crepe_model_capacity","full")).predict,[gr.Audio(type="filepath",label="Input Audio"),gr.Checkbox(False,label="viterbi",info="Apply viterbi smoothing to the estimated pitch curve"),gr.Checkbox(True,label="center"),gr.Number(10,label="step size",info="The step size in milliseconds for running pitch estimation."),gr.Number(0.8,label="minimum confidence"),gr.Number(32,label="batch size")],gr.File(label="Midi File")),
    gr.Interface(CrepeTF(getenv("crepe_model_capacity","full")).predict,[gr.Audio(type="filepath",label="Input Audio"),gr.Checkbox(False,label="viterbi",info="Apply viterbi smoothing to the estimated pitch curve"),gr.Checkbox(True,label="center"),gr.Number(10,label="step size",info="The step size in milliseconds for running pitch estimation."),gr.Number(0.8,label="minimum confidence"),gr.Number(32,label="batch size")],gr.File(label="Midi File")),
    gr.Interface(Pop2Piano(device).predict,[gr.Audio(label="Input Audio",type="filepath"),gr.Number(1, minimum=1, maximum=21, label="Composer"),gr.Number(2,label="Details in Piano"),gr.Number(1,label="Efficiency of Piano"),gr.Radio([1,2,4],label="steps per beat",value=2)],gr.File(label="MIDI File")),
    gr.Interface(YMT3(hf_hub_download("shethjenil/Audio2Midi_Models",f"{getenv('yourmt3_model_type','YMT3+')}.pt"),getenv("yourmt3_model_type","YMT3+"),"32" if str(device) == "cpu" else "16",device).predict,gr.Audio(label="Input Audio",type="filepath"),gr.File(label="MIDI File")),
    midi_viz_ui
],["Normal Pitch Detection","Guitar Based Pitch Detection","Melodia","Spotify Pitch Detection","Magenta Pitch Detection","Violin Based Pitch Detection","Crepe Pitch Detection","Crepe Pitch Detection TF","Pop2Piano","YourMT3","Midi Vizulizer"]).launch()
```
