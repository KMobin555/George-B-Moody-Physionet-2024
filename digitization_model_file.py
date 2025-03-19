import torch
import torch.nn as nn
import torch.nn.functional as F

class DIGITIZATION_MODEL(nn.Module):
    def __init__(self, num_samples, num_leads, img_size=(425, 650), patch_size=16, num_layers=3, num_heads=4, d_model=256, dim_feedforward=2048):
        super(DIGITIZATION_MODEL, self).__init__()
        
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.num_samples = num_samples
        self.num_leads = num_leads
        self.signal_len = num_samples * num_leads
        
        # Patch Embedding
        self.patch_embedding = nn.Linear(patch_size * patch_size, d_model)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # Transformer Encoder and Decoder Layers
        self.transformer_encoder1 = self._create_transformer_encoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_encoder2 = self._create_transformer_encoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_encoder3 = self._create_transformer_encoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_encoder4 = self._create_transformer_encoder_layer(d_model, num_heads, dim_feedforward)
        
        self.bottleneck = self._create_transformer_encoder_layer(d_model, num_heads, dim_feedforward, num_layers=num_layers)
        
        self.transformer_decoder1 = self._create_transformer_decoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_decoder2 = self._create_transformer_decoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_decoder3 = self._create_transformer_decoder_layer(d_model, num_heads, dim_feedforward)
        self.transformer_decoder4 = self._create_transformer_decoder_layer(d_model, num_heads, dim_feedforward)
        
        # Output Layer: Multiple Linear Layers with nn.Sequential
        self.fc = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, self.signal_len)
        )
        
    def _create_transformer_encoder_layer(self, d_model, num_heads, dim_feedforward, num_layers=1):
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _create_transformer_decoder_layer(self, d_model, num_heads, dim_feedforward, num_layers=1):
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        return nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # Patch extraction and embedding
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size)
        x = self.patch_embedding(x)

        # Ensure the patch embeddings fit the number of patches
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), output_size=self.num_patches).permute(0, 2, 1)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer Encoders
        x = self.transformer_encoder1(x)
        x = self.transformer_encoder2(x)
        x = self.transformer_encoder3(x)
        x = self.transformer_encoder4(x)
        
        # Bottleneck Transformer Block
        memory = self.bottleneck(x)
        
        # Transformer Decoders
        x = self.transformer_decoder1(x, memory=memory)
        x = self.transformer_decoder2(x, memory=memory)
        x = self.transformer_decoder3(x, memory=memory)
        x = self.transformer_decoder4(x, memory=memory)
        
        # Output ECG signal using multiple linear layers
        output_signal = self.fc(x.mean(dim=1))
        
        return output_signal
