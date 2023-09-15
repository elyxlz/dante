from azure.storage.blob import BlobServiceClient, ContainerClient
import io
import torchaudio
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np
from typing import Dict
import torch
import logging
import pandas as pd


class IterableWebDataset(IterableDataset):
    def __init__(
        self,
        logger,
        dataset_name,
        account_name,
        account_query_key,
        dev=False
    ):
        
        self.logger = logger
        torchaudio.set_audio_backend('soundfile')

        
        """ Blob Stuff """
        
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

        self.account_name = account_name
        self.account_key = account_query_key

        self.container_clients: Dict[str, ContainerClient] = {}
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=self.account_key
        )
       

        """ Df stuff """
             
        # csv_client = self.blob_service_client.get_container_client('csv')
        # blob_client = csv_client.get_blob_client(dataset_name + '.csv')
        
        # # check if already in cache otherwise download to cache
        # self.logger.info(f"Downloading csv for dataset {dataset_name}")
        # data = io.BytesIO(blob_client.download_blob().readall())
        
        self.df = pd.read_csv("temp.csv") # REMEMBER TO GO ###
        
        if dev:
            self.df = self.df[:10000]
            
        # forgot to tmux, already processed abt 450 podcasts
        self.df = self.df[0:]
                
        self.logger.info(self.df.info())
        
        self.length = len(self.df)
        
        self.logger.info(f"Length of dataset: {self.length}")
        
        containers = self.df['container'].unique()
        
        for container in containers:
            self.container_clients[container] = self.blob_service_client.get_container_client(container)
       

    def __len__(self):
        return self.length
    
    def _get_waveform_from_blob(self, idx: int) -> tuple[Tensor, int]:
        # Load waveform
        try:
            idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
            row = self.df.iloc[idx]
            container = row['container']
            blob_id = row['blob_id']
            container_client = self.container_clients[container]
            blob_client = container_client.get_blob_client(blob_id.strip())
            data = io.BytesIO(blob_client.download_blob().readall())
            if blob_id.endswith('.mp3'):
                waveform, sample_rate = torchaudio.load(data, format='mp3')
            else:
                waveform, sample_rate = torchaudio.load(data)
                
            return dict(
                waveform=waveform, 
                sample_rate=sample_rate,
                container=container,
                blob_id=blob_id
            )
        
        except Exception as e:
            self.logger.error("Error loading file: ", self.df.iloc[idx], e)
            return None