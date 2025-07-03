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
from torch import device as Device
from torch.cuda import is_available as cuda_is_available
device = Device("cuda" if cuda_is_available() else "cpu")
Crepe().predict(audio_path)
Pop2Piano(device=device).predict(audio_path)
Violin_Pitch_Det(device=device).predict(audio_path)
BasicPitch(device=device).predict(audio_path)
```