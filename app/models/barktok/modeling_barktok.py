from typing import Optional, Dict


import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers import BarkCoarseModel, BarkFineModel, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

import os.path
import shutil
import urllib.request
import huggingface_hub
import json
import os.path
from zipfile import ZipFile
import torch
from torch import nn, optim
from torch.serialization import MAP_LOCATION
from pathlib import Path
import torch
from torch import nn
from einops import pack, unpack
import fairseq
from torchaudio.functional import resample
from audiolm_pytorch.utils import curtail_to_multiple
from vocos import Vocos

from .configuration_barktok import BarkTokConfig

class Data:
    input_size: int
    hidden_size: int
    output_size: int
    version: int

    def __init__(self, input_size=768, hidden_size=1024, output_size=10000, version=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    @staticmethod
    def load(string):
        data = json.loads(string)
        return Data(data['input_size'], data['hidden_size'], data['output_size'], data['version'])

    def save(self):
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'version': self.version,
        }
        return json.dumps(data)

class CustomTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, input_size=768, output_size=10000, version=0):
        super(CustomTokenizer, self).__init__()
        next_size = input_size
        if version == 0:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            next_size = hidden_size
        if version == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            self.intermediate = nn.Linear(hidden_size, 4096)
            next_size = 4096

        self.fc = nn.Linear(next_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    @torch.compile
    def forward(self, x):
        x, _ = self.lstm(x)
        if self.version == 1:
            x = self.intermediate(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    @torch.no_grad()
    def get_token(self, x):
        """
        Used to get the token for the first
        :param x: An array with shape (N, input_size) where N is a whole number greater or equal to 1, and input_size is the input size used when creating the model.
        :return: An array with shape (N,) where N is the same as N from the input. Every number in the array is a whole number in range 0...output_size - 1 where output_size is the output size used when creating the model.
        """
        return torch.argmax(self(x), dim=1)

    def save(self, path):
        info_path = '.'.join(os.path.basename(path).split('.')[:-1]) + '/.info'
        torch.save(self.state_dict(), path)
        data_from_model = Data(self.input_size, self.hidden_size, self.output_size, self.version)
        with ZipFile(path, 'a') as model_zip:
            model_zip.writestr(info_path, data_from_model.save())
            model_zip.close()

    @staticmethod
    def load_from_checkpoint(path, map_location: MAP_LOCATION = None):
        old = True
        with ZipFile(path) as model_zip:
            filesMatch = [file for file in model_zip.namelist() if file.endswith('/.info')]
            file = filesMatch[0] if filesMatch else None
            if file:
                old = False
                data_from_model = Data.load(model_zip.read(file).decode('utf-8'))
            model_zip.close()
        if old:
            model = CustomTokenizer()
        else:
            model = CustomTokenizer(data_from_model.hidden_size, data_from_model.input_size, data_from_model.output_size, data_from_model.version)
        model.load_state_dict(torch.load(path))
        if map_location:
            model = model.to(map_location)
        return model


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class CustomHubert(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        target_sample_hz=24000,
        seq_len_multiple_of=None,
        output_layer=9,
        device=None
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        if device is not None:
            self.to(device)

        model_path = Path(checkpoint_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        if device is not None:
            model[0].to(device)

        self.model = model[0]
        self.model.eval()

    @property
    def groups(self):
        return 1

    @torch.inference_mode()
    def forward(
        self,
        wav_input,
        flatten=False,
        input_sample_hz=None
    ):
        device = wav_input.device

        embed = self.model(
            wav_input,
            features_only=True,
            mask=False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
            output_layer=self.output_layer
        )

        embed, packed_shape = pack([embed['x']], '* d')
        #embed = embed['x']
        #print(embed.shape)
        #return embed

        # codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        #codebook_indices = torch.from_numpy(embed.cpu().detach().numpy()).to(device)  # .long()

        if flatten: 
            return embed

        embed, = unpack(embed, packed_shape, '*')
        return embed


class HuBERTManager:
    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', file_name: str = 'hubert.pt'):
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, file_name)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')
        return install_file


    @staticmethod
    def make_sure_tokenizer_installed(model: str = 'quantifier_hubert_base_ls960_14.pth', repo: str = 'GitMylo/bark-voice-cloning', local_file: str = 'tokenizer.pth'):
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, local_file)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, model, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, model), install_file)
            print('Downloaded tokenizer')
        return install_file
    

class MyBarkModel(BarkModel):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def generate(
        self,
        z: Optional[torch.Tensor] = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.LongTensor:

        # TODO (joao):workaround until nested generation config is compatible with PreTrained Model
        # todo: dict
        semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
        coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
        fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)

        # 2. Generate from the coarse model
        coarse_output = self.coarse_acoustics.generate(
            z,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=self.generation_config.codebook_size,
            #**kwargs_coarse,
        )

        # 3. "generate" from the fine model
        output = self.fine_acoustics.generate(
            coarse_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            fine_generation_config=fine_generation_config,
            codebook_size=self.generation_config.codebook_size,
            #**kwargs_fine,
        )

        return output


class BarkTok(PreTrainedModel):
    def __init__(self, config: BarkTokConfig = None):
        
        if config is None:
            config = BarkTokConfig()
        
        super().__init__(config)
                
        self.hubert_manager = HuBERTManager()
        self.hubert_manager.make_sure_hubert_installed()
        self.hubert_manager.make_sure_tokenizer_installed()
        
        self.hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt')
        self.ubert_model = torch.compile(self.hubert_model)


        # Load the CustomTokenizer model
        self.tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth')  # Automatically uses the right layers

        self.coarse_model: BarkCoarseModel = BarkCoarseModel.from_pretrained("suno/bark")
        self.fine_model: BarkFineModel = BarkFineModel.from_pretrained("suno/bark")

        self.bark_model: MyBarkModel = MyBarkModel.from_pretrained("suno/bark")
        
        #self.device = next(self.parameters()).device
        self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")#.to(self.device)

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, f"Expected 3 dimensions, got {x.dim()}"
        b = x.shape[0]
        x = x.mean(1)
        semantic_vectors = self.hubert_model.forward(x, input_sample_hz=16000, flatten=True)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors).view(b, -1)
        return semantic_tokens
        
    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        d = z.device
        codes = self.bark_model.generate(z).permute(1, 0, 2)
        features = self.vocos.codes_to_features(codes)
        bandwidth_id = torch.tensor([2]).to(d)  # 6 kbps
        out = self.vocos.decode(features, bandwidth_id=bandwidth_id).unsqueeze(1)
        return out




