import telebot as tb
from omegaconf import OmegaConf
import soundfile as sf
import io
import argparse as ap
import numpy as np

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
dante: LlamaForCausalLM = model.merge_and_unload()

tokenizer: BarkTok = BarkTok().cuda()


bot_token = ""
bot = tb.TeleBot(
    token=bot_token,
)
bot.set_my_name(
    "Dante",
)



    
@bot.message_handler(content_types=['voice'], chat_types=['private'])
def voice(message: tb.types.Message):

    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    data, sample_rate = sf.read(io.BytesIO(downloaded_file))
    data = data.astype(np.float32)
    data = torch.from_numpy(data).cuda()
    
    
    data = ta.functional.resample(data, sample_rate, 16000).cuda().unsqueeze(0)
    prompt_tokens = tokenizer.encode(data)
    # if longer than 350 cut front
    if prompt_tokens.shape[1] > 350:
        prompt_tokens = prompt_tokens[..., -350:]
                
    generate_kwargs = {
        "do_sample": True,
        "max_length": 750,
        "temperature": 1.,
        "top_k": 20,
        "top_p": 0.1,
        "attention_mask": torch.ones_like(prompt_tokens),
        "use_cache": True,
    }
    
    continuation_tokens = dante.generate(
        inputs=prompt_tokens,
        **generate_kwargs
    )
    
    bot.send_chat_action(message.chat.id, 'record_audio')
    
    continuation_audio = tokenizer.decode(continuation_tokens).cpu().numpy()[0]
    
    buffer = io.BytesIO()
    sf.write(buffer, continuation_audio, 24000, format='ogg')
    buffer.name = 'test.mp3'
    buffer.seek(0)
    
    message_id = int(str(message.chat.id) + str(message.message_id))
    
    


    reply = bot.send_voice(message.chat.id, buffer)
    reply_id = int(str(bot.get_me().id) + str(reply.message_id))
    


if __name__ == '__main__':
    bot.infinity_polling()
