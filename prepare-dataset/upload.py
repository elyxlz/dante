from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
import os



def azure_upload_filelike(
    connection_string,
    container,
    buffer,
    filename
):
    # Set the name of the container you want to create or use

    credential = DefaultAzureCredential()

    if (connection_string != None):
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string)
    else:
        raise Exception("No connection string or account url provided")

    container_client = blob_service_client.get_container_client(container)
    try:
        container_client.create_container()
    except:
        # Container already exists
        pass
    

    # Get a reference to the BlobClient for the file
    blob_client = blob_service_client.get_blob_client(
        container=container, blob=filename)

    # Upload the file to Azure Blob Storage
    try:
        blob_client.upload_blob(buffer, overwrite=True)
        return True
    
    except Exception as e:
        print("Failed to upload on azure:")
        print(e)
        return False
    
    