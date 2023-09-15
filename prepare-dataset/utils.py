from torch import Tensor, nn
import torch
import torchaudio
import einops
import torchaudio
import logging
import colorlog

def validate_audio(waveform: Tensor, sample_rate: int):
    # check sample rate is even
    if sample_rate % 2 != 0:
        return False
        
    # if extremely short
    if waveform.shape[-1] < 5:
        return False
                    
    # if more than two channels
    if waveform.shape[0] > 2:
        return False
    
    return True 
                               
def is_silence(waveform: torch.Tensor, thresh: int = -35):
    waveform = torch.mean(waveform, dim=0)
    dBmax = 20 * torch.log10(torch.abs(waveform + 1e-6)).max().flatten()
    return dBmax < thresh

def leading_silence(audio: torch.Tensor, sample_rate: int, thresh=-35):
    chunk_size = int(sample_rate / 100) # 10ms
    trim_ms = 0
    while is_silence(audio[:, trim_ms:trim_ms+chunk_size], thresh=thresh) and trim_ms < audio.shape[0]:
        trim_ms += chunk_size
    return trim_ms


# logger

def setup_logger(logging_level: str):
    # Step 1: Configure the root logger and remove default handler
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.ERROR)

    # Step 2: Create your custom logger
    logger = logging.getLogger("__name__")
    logger.setLevel(getattr(logging, logging_level.upper()))

    # Step 3: Create a colored log handler
    log_handler = colorlog.StreamHandler()
    log_handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    # Step 4: Add the colored handler to your custom logger
    logger.addHandler(log_handler)

    return logger