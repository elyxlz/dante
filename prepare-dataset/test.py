from models.barktok.modeling_barktok import BarkTok
import torchaudio as ta
import torch

tokenizer = BarkTok()
audio = ta.load("test.wav")[0]
audio = ta.functional.resample(audio, 48000, 16000).unsqueeze(0)

print(audio.shape)
tokens = tokenizer.encode(audio)
print(tokens.shape)
recon = tokenizer.decode(tokens)
import pdb; pdb.set_trace()
ta.save("recon.wav", recon.cpu()[0], 24000)

