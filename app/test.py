# lets test out inference by getting a pretrained model, tokenizing this test.wav as a prompt, continuing
# and detokenizing

from transformers import LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from models.barktok.modeling_barktok import BarkTok
import os
import torchaudio as ta
import torch

import dotenv
dotenv.load_dotenv()

bnb_config = BitsAndBytesConfig(
    load_in8bit=False
)

pretrain = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="cuda:0",
    quantization_config=bnb_config,
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]    
)

model: PeftModel = PeftModel.from_pretrained(pretrain, "Audiogen/dante-hackathon-630").cuda()
model: LlamaForCausalLM = model.merge_and_unload()

tokenizer: BarkTok = BarkTok().cuda()

audio_prompt = ta.load("test.wav")[0].cuda()
audio_prompt = ta.functional.resample(audio_prompt, 48000, 16000).cuda().unsqueeze(0)
audio_prompt = tokenizer.encode(audio_prompt)[..., :150]
print(audio_prompt.shape)

def generate(prompt, decode=False, **kwargs):
    generate_kwargs = {
        "do_sample": True,
        "max_length": 500,
        "temperature": 1.,
        "top_k": 20,
        "top_p": 0.1,
        "attention_mask": torch.ones_like(prompt),
        "use_cache": True,
    }

    # update generate_kwargs with kwargs
    generate_kwargs.update(kwargs)

    with torch.no_grad():
        continuation = model.generate(
            inputs=prompt,
            **generate_kwargs
        )
        
    if decode:
        continuation = tokenizer.decode(continuation)
        ta.save("continue.wav", continuation[0].cpu(), 24000)
    
    return continuation

continuation = generate(audio_prompt)
import pdb; pdb.set_trace()

print("decoding")
continuation_audio = tokenizer.decode(continuation)
ta.save("continue.wav", continuation_audio[0].cpu(), 24000)

import pdb; pdb.set_trace()

