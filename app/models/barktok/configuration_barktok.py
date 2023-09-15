from transformers import PretrainedConfig

class BarkTokConfig(PretrainedConfig):
    model_type = "barktok"
    
    def __init__(
        self,
        **kwargs
    ):
        
        super().__init__(**kwargs)