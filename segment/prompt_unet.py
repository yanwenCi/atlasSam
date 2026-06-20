import torch
import torch.nn as nn
import torch.nn.functional as F

# === Define 3D U-Net Encoder ===
class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_layers=5):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        # Create encoder layers dynamically
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else base_channels * (2**(i-1))
            out_ch = base_channels * (2**i)
            self.encoders.append(nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                                               nn.BatchNorm3d(out_ch),
                                               ))
            self.pools.append(nn.MaxPool3d(2))

        # Neck layer (bottleneck)
        self.neck = nn.Conv3d(base_channels * (2**(num_layers-1)), base_channels * (2**num_layers), kernel_size=3, padding=1)

    def forward(self, x):
        skip_connections = []
        for i, (conv, pool) in enumerate(zip(self.encoders, self.pools + [None])):
            x = F.relu(conv(x))
            if pool:
                skip_connections.append(x)  # Save for U-Net skip connections
                x = pool(x)  # Downsample

        x = F.relu(self.neck(x))  # Neck/bottleneck layer
        return x, skip_connections  # Reverse skip connections for decoder


# === Define Mask Encoder ===
class MaskEncoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=4, stride=4),
            

          
        )
    def forward(self, mask):
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)  # Add channel dimension
        return self.conv(mask)

# === Define 3D U-Net Decoder with Mask Fusion ===
class UNet3DDecoder(nn.Module):
    def __init__(self, base_channels=32, out_channels=1, num_layers=4, latent_dim=64):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(num_layers-1, -1, -1):
            input_ch = base_channels * (2 ** (i + 1)) 
            output_ch = base_channels * (2 ** i)

            if i == num_layers - 1:
                self.upconvs.append(
                    nn.ConvTranspose3d(input_ch+latent_dim, output_ch, kernel_size=2, stride=2)
                )
            else:
                self.upconvs.append(
                nn.ConvTranspose3d(input_ch, output_ch, kernel_size=2, stride=2)
                ) 
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv3d(input_ch, output_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(output_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(output_ch, output_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(output_ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, encoder_features, mask_embedding):
        x = torch.cat([x, mask_embedding], dim=1) if mask_embedding is not None else x
        for i in range(len(self.upconvs)):
            
            x = self.upconvs[i](x)
            # Align shapes if necessary (can happen due to pooling rounding)
            if x.shape != encoder_features[-(i + 1)].shape:
                x = F.interpolate(x, size=encoder_features[-(i + 1)].shape[2:], mode="trilinear", align_corners=False)
            # print(i, x.shape, encoder_features[-(i + 1)].shape)
            x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
            x = self.dec_blocks[i](x)
        return self.final_conv(x)


# === Combine Modules into Full Model ===
class PromptedUNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, latent_dim=64, out_channels=1, num_layers=5):
        super().__init__()
        self.encoder = UNet3DEncoder(in_channels, base_channels, num_layers)
        self.mask_encoder = MaskEncoder3D(1, latent_dim)
        self.decoder = UNet3DDecoder(base_channels,  out_channels, num_layers, latent_dim,)

    def forward(self, image, mask):
        image_features, skip_features = self.encoder(image)  # Extract features
        mask_embedding = self.mask_encoder(mask)  # Encode mask prompt
        segmentation = self.decoder(image_features, skip_features, mask_embedding)  # Decode with fusion
        return segmentation

# === Example Usage ===
if __name__ == '__main__':
    model = PromptedUNet3D()
    image = torch.randn(1, 1, 192, 192, 96)  # Example 3D volume (batch, channels, depth, height, width)
    mask = torch.randn(1, 1, 192//4, 192//4, 96//4)  # Example 3D mask
    output = model(image, mask)
    print(output.shape)  # Should be (1, 1, 64, 128, 128)
