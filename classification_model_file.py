import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(MinimalPatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, grid_size[0], grid_size[1]]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x

class MinimalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim):
        super(MinimalTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            batch_first = True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, embed_dim)
        return x

class MinimalViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, hidden_dim, num_classes):
        super(MinimalViT, self).__init__()
        self.patch_embed = MinimalPatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = MinimalTransformerEncoder(embed_dim, num_heads, num_layers, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]  # Use the output corresponding to the [CLS] token
        x = self.fc(cls_output)
        return x

class MinimalSignalTransformer(nn.Module):
    def __init__(self, signal_len, embed_dim=16, num_heads=2, num_layers=1, hidden_dim=32):
        super(MinimalSignalTransformer, self).__init__()
        self.embed = nn.Linear(1, embed_dim)  # Embedding for 1D signal input
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, 32)  # Adjust output dimension

    def forward(self, signal):
        signal = signal.unsqueeze(-1)  # Add channel dimension
        embedded_signal = self.embed(signal)
        transformer_output = self.transformer_encoder(embedded_signal.permute(1, 0, 2))  # Permute for Transformer: [seq_len, batch, embed_dim]
        pooled_output = transformer_output.mean(dim=0)  # Pooling over the sequence length
        signal_features = self.fc(pooled_output)
        return signal_features

class CLASSIFICATION_MODEL(nn.Module):
    def __init__(self, list_of_classes, signal_len, img_size=(425, 650)):
        super(CLASSIFICATION_MODEL, self).__init__()
        
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        
        # Image and Signal Processing Modules
        self.image_processor = MinimalViT(
            img_size=img_size,
            patch_size=(32, 32),  # Larger patch size for simplicity
            in_channels=3,
            embed_dim=128,
            num_heads=2,
            num_layers=2,
            hidden_dim=256,
            num_classes=self.num_classes
        )
        
        self.signal_processor = MinimalSignalTransformer(signal_len)
        
        # Fully connected layers
        self.fc1 = nn.Linear(11, 128)  # Adjusted size
        # self.fc2 = nn.Linear(128, self.num_classes)  # Output layer for classification
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_classes)
        )
    
    def forward(self, image, signal):
        image_features = self.image_processor(image)
        # signal_features = self.signal_processor(signal)

        # combined_features = torch.cat((image_features, signal_features), dim=1) 
        
        output = F.relu(self.fc1(image_features))
        output = torch.sigmoid(self.fc2(output))  # Sigmoid activation for multilabel classification
        
        return output
