import torch
import torch.nn as nn
import torch.nn.functional as F

class DIGITIZATION_MODEL(nn.Module):
    def __init__(self, num_samples=5000, num_leads=12):
        super(DIGITIZATION_MODEL, self).__init__()
        
        self.num_samples = num_samples
        self.num_leads = num_leads

        # Encoder
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Output layer
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

        # LSTM
        self.rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024,4096)

        self.fc = nn.Linear(4096, self.num_samples*self.num_leads)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        
        # Adjust dimensions if necessary
        if dec4.size()[2:] != enc4.size()[2:]:
            dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=False)
            
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        
        # Adjust dimensions if necessary
        if dec3.size()[2:] != enc3.size()[2:]:
            dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=False)
            
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        
        # Adjust dimensions if necessary
        if dec2.size()[2:] != enc2.size()[2:]:
            dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=False)
            
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        
        # Adjust dimensions if necessary
        if dec1.size()[2:] != enc1.size()[2:]:
            dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=False)
            
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.conv_last(dec1)

        # Flatten the output and pass through LSTM
        batch_size = output.size(0)
        output = F.adaptive_avg_pool2d(output, (1, 256)).view(batch_size, 1, 256)
        rnn_out, _ = self.rnn(output)
        output = rnn_out[:, -1, :]

        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        
        output = self.fc(output)
        
        return output