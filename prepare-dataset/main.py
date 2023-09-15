import torch 
from torch import nn
import torchaudio
from einops import rearrange
from torch.utils.data import DataLoader
import transformers as hf
import wandb

from argparse import ArgumentParser
from tqdm import tqdm
from omegaconf import OmegaConf
import threading
import concurrent.futures

import io
import os
import time
import logging

import dotenv
dotenv.load_dotenv()

from webdataset import IterableWebDataset
from upload import azure_upload_filelike
from utils import validate_audio, is_silence, leading_silence, setup_logger

import tracemalloc

class CPUProcessing(IterableWebDataset):
    """
    This class is for processing the raw data on cpu.j
    It removes corrupt data. Converts it to the proper sample rate and channel num.
    Chunks it, and only returns non-silent chunks with no leading silence.
    Iterable instead of map style dataset for 2 reasons:
        1. Survives uneven raw data sizes, e.g. 2 minute song or 10 hour podcast doesn't slow down dataloading.
        2. Allows fixed batch sizes regardless of the number of chunks in a file.
    
    This dataset is fed an index to a bunch of blobs, these indeces are placed in a queue which is completed via multi-threading,
    the chunks therefore are "streamed" to the dataloader for gpu processing as they are processed on cpu.
    
    Args:
        logger (logging.Logger): The logger to use.
        channels (int): The number of channels to convert to.
        chunk_length (int): The size of the chunks to return in seconds. If 0, no chunking is done.
        remove_leading_silence (bool): Whether to remove initial silence from the audio.
        silence_threshold (int): The threshold for silence detection.
        dataset_name (str): The name of the dataset.
        account_name (str): The name of the azure account.
        account_query_key (str): The query key for the azure account.
    
    Returns:
        dict: A dictionary containing the chunks.
    
    """
    def __init__(
        self,
        logger: logging.Logger,
        channels: int,
        chunk_length: int, # in seconds 
        remove_leading_silence: bool,
        silence_threshold: int,
        dataset_name: str,
        account_name: str,
        account_query_key: str,
        **kwargs
    ):
        self.logger = logger
        self.channels = channels
        self.chunk_length = chunk_length
        self.remove_leading_silence = remove_leading_silence
        self.silence_threshold = silence_threshold
        
        super().__init__(
            logger=logger,
            dataset_name=dataset_name,
            account_name=account_name,
            account_query_key=account_query_key,
            **kwargs
        )
        
        self.max_idx = len(self)
        
        self.task_queue = torch.multiprocessing.Queue()
        self.lock = torch.multiprocessing.Lock()
        self.pbar = tqdm(total=self.max_idx, desc="Raw files processed", colour='green')


        with self.lock:
            for idx in range(self.max_idx):
                self.task_queue.put(idx)
            
               
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        while True:
            with self.lock:
                try:
                    idx = self.task_queue.get_nowait()
                    #idx = torch.randint(0, self.max_idx, (1,)).item()
                except Exception as e:
                    print(f"Worker {worker_id} finished", e)
                    break  # All tasks are completed


            out = self._get_waveform_from_blob(idx)
            
            if out is None:
                continue
            
            waveform = out['waveform']
            sample_rate = out['sample_rate']
            channels = waveform.shape[0]
            
            # validate
            if not validate_audio(waveform, sample_rate):
                self.logger.info(f"File {idx} is corrupt")
                continue
            
            # resample
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                
            # re-channel
            if channels != self.channels:
                if channels == 1 and self.channels == 2:
                    waveform = waveform.repeat(2, 1)
                else:
                    waveform = waveform.mean(0, keepdim=True)
            
            # chunk
            if self.chunk_length > 0:
                frames = int(self.chunk_length * 16000)
                chunks = torch.split(waveform, frames, dim=-1)
                
                for chunk in chunks:
                                        
                    out = dict()
                    
                    # check if not too short
                    if chunk.shape[-1] < 10000:
                        continue
                    
                    # check if not silent
                    if is_silence(chunk):
                        continue
                    
                    # remove leading silence
                    if self.remove_leading_silence:
                        chunk = chunk[:, leading_silence(chunk, 16000, self.silence_threshold):]
                    
                    # pad to chunk size
                    if chunk.shape[-1] < frames:
                        chunk = torch.nn.functional.pad(chunk, (0, frames - chunk.shape[-1]))
                                          
                    out['audio_chunks'] = chunk
                    yield out
                    
            # update pbar
            if worker_id == 0:
                with self.lock:
                    self.pbar.n = self.max_idx - self.task_queue.qsize()
                    self.pbar.refresh()
                                        
                        
                        
class GPUProcessing(nn.Module):
    """
    This class is for processing the data on gpu
    It is also for uploading the data to blob.
    
    Args:
        name (str): The name of the target dataset.
        logger (logging.Logger): The logger.
        ds (Dataset): The initialised dataset object.
        batch_size (int): The batch size to use for processing.
        num_workers (int): The number of workers to use for processing.
    """
    
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        ds: CPUProcessing,
        sample_rate: int,
        batch_size: int,
        num_workers: int,
        torch_compile: bool = False,
    ):
        super().__init__()
        
        self.name = name
        self.logger = logger
        
        self.ds = ds
        self.dl = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if self.device == 'cpu':
            self.logger.warning("GPUProcessing initialised on CPU")


        from models.barktok.modeling_barktok import BarkTok
        self.vqvae = BarkTok().to(self.device)
       
        
        # forgot to tmux, resume progress, abt 142000 files processed
        self.upload_idx = 0
        self.upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=512)
        self.lock = threading.Lock()
        
        self.logger.info(f"GPUProcessing initialised on {self.device}")
        

    def _upload_worker(self, out, filename):
        buffer = io.BytesIO()
        torch.save(out, buffer)
        buffer.seek(0)
            
        success = azure_upload_filelike(
            connection_string=os.environ['BLOB_CONNECTION_STRING'],
            container=self.name,
            buffer=buffer,
            filename=filename,
        )

        
        # check if success
        if success:
            #with self.lock:
            #    self.upload_idx += 1
            #print(f"uploaded {filename}")
            pass
        else:
            self.logger.error(f"Failed to upload {filename}", success)
            
    # def _process_and_upload(self, out_list):
    #     for out in out_list:
    #         with self.lock:
    #             filename = f"{self.upload_idx}.pt"
    #         self.upload_executor.submit(self._upload_worker, out, filename)
            
                    
    @torch.no_grad()
    def forward(self):
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
        
            for i, batch in tqdm(enumerate(self.dl), position=0):
                self.logger.debug(f"Processing batch {i} of size {batch['audio_chunks'].shape[0]}")
                
                audio_chunks = batch['audio_chunks'].to(self.device)
                                                
                # vqvae
                self.logger.debug("Extracting vqvae tokens")
                tokens = self.vqvae.encode(audio_chunks)
                out_list = tokens.unbind(0)
         
                self.logger.debug("Starting upload")
                # put this in a new thread
                #threading.Thread(target=self._process_and_upload, args=(out_list,)).start()
                idxs = range(i * self.batch_size, i * self.batch_size + self.batch_size)
                for n, out in enumerate(out_list):
                    out = out.clone().cpu()
                    filename = f"{idxs[n]}.pt"
                    self.upload_executor.submit(self._upload_worker, out, filename)
                            
                self.logger.debug(f"Uploaded {len(out_list)} files.")
             

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('config', type=str)
    args = args.parse_args()
    cfg = OmegaConf.load(os.path.join('./configs', f"{args.config}.yaml"))
    
    logging_level = cfg.params.logging_level
    logger = setup_logger(logging_level)
    
    if cfg.params.wandb:
        wandb.init(
            project="prepare-dataset",
            name=cfg.name,
            config=OmegaConf.to_container(cfg),
        )
    
    ds = CPUProcessing(
        logger=logger,
        channels=cfg.processing.channels,
        chunk_length=cfg.processing.chunk_length,
        remove_leading_silence=cfg.processing.remove_leading_silence,
        silence_threshold=cfg.processing.silence_threshold,
        dataset_name=cfg.raw_data.dataset_name,
        account_name=cfg.raw_data.account_name,
        account_query_key=os.environ['BLOB_KEY'],
    )
    

    main = GPUProcessing(
        name=cfg.name,
        logger=logger,
        ds=ds,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        sample_rate=cfg.processing.sample_rate,
    )
    
    main()
        
    if cfg.params.wandb:
        wandb.finish()
        
        
        