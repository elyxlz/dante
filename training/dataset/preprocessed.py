from azure.storage.blob import BlobServiceClient, ContainerClient
import io # type: ignore
import os
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict # type: ignore
import torch
import logging
import pandas as pd
from tqdm import tqdm



class PreProcessedDataset(Dataset):
    """
    Map style dataset for preprocessed .pt files.
    Right now requires the max length of the dataset to build index, thinking of ways to fix this.
    
    Args:
        dataset_name: name of dataset
        dataset_length: number of rows in dataset, needed to create index
        account_name: name of account
        dev: if True, only loads 10k rows of dataset
    
    """
    def __init__(
        self,
        dataset_name,
        dataset_length,
        account_name,
        dev=False
    ):  
        
        """ Blob Stuff """
        
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

        self.account_name = account_name
        self.account_key = os.environ['BLOB_KEY']

        self.container_clients: Dict[str, ContainerClient] = {}
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=self.account_key
        )
       

        """ Df stuff """
        
        self.index = torch.arange(dataset_length)
                
        if dev:
            self.index = self.idx[:10000]
                                
        self.container_client = self.blob_service_client.get_container_client(dataset_name)

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, int]: # type: ignore
        # Load waveform
        try:
            idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
            blob_client = self.container_client.get_blob_client(f"{idx}.pt")
            data = io.BytesIO(blob_client.download_blob().readall())
            input_ids = torch.load(data)
            #input_ids = torch.randint_like(input_ids, 0, 200) # TODO: REMOVE
            labels = input_ids[1:].clone().contiguous()
            labels = torch.cat([labels, -100 * torch.ones((1,), dtype=torch.long)], dim=0)
            attention_mask = torch.ones_like(input_ids)
            
            out = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            return out
        
        except Exception as e:
            print("Error loading file: ", idx, e)
            idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(idx)
