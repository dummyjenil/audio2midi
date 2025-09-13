from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor
from librosa import load as librosa_load
from huggingface_hub import snapshot_download

class Pop2Piano:
    def __init__(self,device="cpu",model_path=None):
        if not model_path:
            model_path=snapshot_download("sweetcocoa/pop2piano")
        self.model = Pop2PianoForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = Pop2PianoProcessor.from_pretrained(model_path)

    def predict(self,audio,composer=1,num_bars=2,num_beams=1,steps_per_beat=2,output_file="output.mid"):
        data, sr = librosa_load(audio, sr=None)
        inputs = self.processor(data, sr, steps_per_beat,return_tensors="pt",num_bars=num_bars)
        self.processor.batch_decode(self.model.generate(num_beams=num_beams,do_sample=True,input_features=inputs["input_features"], composer="composer" + str(composer)),inputs)["pretty_midi_objects"][0].write(open(output_file, "wb"))
        return output_file
