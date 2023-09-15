from typing import Dict

from torch import Tensor
from transformers import LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
import torch
import os
import dotenv
from models.barktok.modeling_barktok import BarkTok
from trainers.base import BaseModule, BaseEvaluation
from dataset.preprocessed import PreProcessedDataset

import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

dotenv.load_dotenv()



# A training script for Llama that uses multi gpu training and PEFT for qlora



# load llama in flash attn mode
model_name = "meta-llama/Llama-2-7b-hf"
#model_name = "BlackSamorez/llama-2-tiny-testing"

# config = AutoConfig.from_pretrained(model_name, use_auth_token=os.environ['HUGGINGFACE_TOKEN'])

# # loading with flash attn
# use_flash_llama = True
# if use_flash_llama and config.model_type == 'llama':
#     updates = {}
#     # this is a fork of togethercomputer/LLaMA-2-7B-32K's modeling_flash_llama.py, with a padding fix
#     # https://huggingface.co/Birchlabs/flash_llama/blob/main/modeling_flash_llama.py
#     flash_model_name = 'Birchlabs/flash_llama--modeling_flash_llama.LlamaForCausalLM'
#     if 'auto_map' in config.__dict__:
#         if not ('AutoModelForCausalLM' in config.auto_map and 'flash' in config.auto_map['AutoModelForCausalLM']):
#             updates['auto_map']['AutoModelForCausalLM'] = flash_model_name
#     else:
#         updates['auto_map'] = { 'AutoModelForCausalLM': flash_model_name }
#     # modeling_flash_llama.py expects some llama 2 config to be present. if this is a llama 1 model: we add the missing config
#     if 'num_key_value_heads' not in config.__dict__:
#         updates['num_key_value_heads'] = config.num_attention_heads
#     if 'rope_scaling' not in config.__dict__:
#         # if you want to finetune to a non-native context length, here's where you'd override it
#         # updates['rope_scaling'] = { 'factor': context_length/config.max_position_embeddings, 'type': 'linear' }
#         updates['rope_scaling'] = None
#     if 'pretraining_tp' not in config.__dict__:
#         updates['pretraining_tp'] = 1
#     if updates:
#         config.update(updates)


########################################################################################


# from typing import List, Optional, Tuple, Union
# import logging

# import torch
# from torch import nn

# import transformers
# from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# from einops import rearrange
# from flash_attn import flash_attn_func



# def forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     bsz, q_len, _ = hidden_states.size()

#     if self.config.pretraining_tp > 1:
#         raise ValueError("pretraining_tp > 1 is not supported for flash attention")
#     else:
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]

#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     query_states, key_states, value_states = [
#         rearrange(x, "b h s d -> b s h d") for x in [query_states, key_states, value_states]
#     ]

#     query_states, key_states, value_states = [x.to(torch.bfloat16) for x in [query_states, key_states, value_states]]
#     # print(f"{query.shape=} {key.shape=} {value.shape=}")
#     # below output will have shape (batch_size, seqlen, nheads, headdim)
#     attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)

#     if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
#         raise ValueError(
#             f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
#             f" {attn_output.size()}"
#         )

#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
#     attn_output = self.o_proj(attn_output)
#     if output_attentions:
#         raise NotImplementedError("`output_attentions` is not supported when `use_flash_attn` is True")
#     attn_weights = None

#     return attn_output, attn_weights, past_key_value


# # Disable the transformation of the attention mask in LlamaModel as the flash attention
# # requires the attention mask to be the same as the key_padding_mask
# def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
#     # [bsz, seq_len]
#     return attention_mask

# transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
#     _prepare_decoder_attention_mask
# )
# transformers.models.llama.modeling_llama.LlamaAttention.forward = forward

######################################################################################
        
# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #device_map="auto",
    device_map={"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))},
    #config=config,
    trust_remote_code=True,
    quantization_config=bnb_config,
    use_auth_token=os.environ['HUGGINGFACE_TOKEN'],
)


model = prepare_model_for_kbit_training(model)

# lora config, getting all module names, attention and linears
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules.extend(list(set(names)))

lora_config = LoraConfig(
    r=32,
    target_modules = target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#tokenizer: BarkTok = BarkTok()

dataset = PreProcessedDataset(
    dataset_name="dante-podcasts-processed",
    dataset_length=120000,
    account_name="audiogentrainingdataeun"
)

training_args = transformers.TrainingArguments(
    #hub_model_id="Audiogen/dante-llama13b-qlora-hackathon",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=8,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    #auto_find_batch_size=True,
    gradient_accumulation_steps=1,
    num_train_epochs=100000,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=4,
    logging_steps=5,
    output_dir='~/training/logs',
    save_strategy='steps',
    save_steps=70,
    evaluation_strategy="steps",
    eval_steps=70,
    run_name="dante",
    #push_to_hub=True,
    #hub_token=os.environ['HUGGINGFACE_TOKEN'],
    torch_compile=False,
)


class MyTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)
    
# split dataset
len_dataset = len(dataset)
split = 0.005
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [int(len_dataset * (1 - split)), int(len_dataset * split)])

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


if __name__ == "__main__":
    trainer.train()



# class Evaluation(BaseEvaluation):
    
#     def __init__(
#         self,
#         **kwargs
#     ):
#         super().__init__(**kwargs)


#     @torch.no_grad()
#     def evaluate(self, pl_module: Module, accum_batch) -> None:
#         audio_chunks = accum_batch['audio_chunks']

#         # split chunks in two, with a randomgit center between 25% and 75% of length

#         b = audio_chunks.shape[0]
#         t = audio_chunks.shape[-1]
        
#         center = torch.randint(int(t * 0.25), int(t * 0.75), (1,)).item()
        
#         chunks0 = audio_chunks[:, :, :center]
#         chunks1 = audio_chunks[:, :, center:]
        
#         tokens0 = pl_module.tokenizer.encode(chunks0)
        
#         continuation_gen = pl_module.model.generate(
#             tokens0,
#             do_sample=True,
#             max_length=int((t - center) // 24000 * 49.9),
#             temperature=1.0,
#             top_k=250,
#             top_p=0.0,
#             attention_mask=torch.ones_like(tokens0),
#             use_cache=True,  
#         )
        
#         # remove all tokens above 10048, make them 0
#         continuation_gen[continuation_gen > 10048] = 10048
        
#         continuation_gen = pl_module.tokenizer.decode(continuation_gen)
                    
#         self.log_wandb_audio(
#             elements=dict(
#                 prompt=chunks0,
#                 continuation_gen=continuation_gen,
#                 continuation_true=chunks1,
#             ),
#             sample_rate=32000,
#             log_mel=True,
#             name="continuations",
#         )