import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn

import dinov2.distributed as distributed
from dinov2.eval.utils import is_padded_matrix, Model3DWrapper
from torchvision.transforms import transforms

class DINOV2Encoder(torch.nn.Module):
    def __init__(self, encoder, autocast_ctx, n_last_blocks=1, is_3d=False) -> None:
        super(DINOV2Encoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.autocast_ctx = autocast_ctx
        self.is_3d = is_3d
        self.n_last_blocks = n_last_blocks
    
    def forward_3d(self, x):
        batch_features = [] 
        for batch_scans in x: # calculate the features for every scan in all scans of the batch
            scans = []
            for scan in batch_scans:
                if not is_padded_matrix(scan): scans.append(self.forward_(scan.unsqueeze(0)))
            batch_features.append(scans)
        return batch_features

    def forward_(self, x):
        with torch.no_grad():
            with self.autocast_ctx():
                if self.n_last_blocks == 1:
                    features = self.encoder.forward_features(x)['x_norm_patchtokens']
                else:
                    features = self.encoder.get_intermediate_layers(
                        x, self.n_last_blocks, return_class_token=False
            )
        return features

    def forward(self, x):
        if self.is_3d:
            return self.forward_3d(x)
        return self.forward_(x)

class LinearDecoder(torch.nn.Module):
    """Linear decoder head"""
    DECODER_TYPE = "linear"

    def __init__(self, in_channels, num_classes=3, image_size=448, patch_size=14):
        super().__init__()
        print(patch_size)
        self.image_size = image_size
        self.in_channels = in_channels
        self.width = self.height = image_size // patch_size
        self.decoder = torch.nn.Conv2d(in_channels, num_classes, (1,1))
        self.decoder.weight.data.normal_(mean=0.0, std=0.01)
        self.decoder.bias.data.zero_()
    
    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        output = self.decoder(embeddings)
        
        # Upsample (interpolate) output/logit map.
        output = torch.nn.functional.interpolate(output, size=self.image_size, mode="bilinear", align_corners=False)

        return output
    
class UNetDecoderUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=1024) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )        

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x2 = self.skip_conv(x2)
        scale_factor = (x1.size()[2] / x2.size()[2])
        x2 = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)(x2)
        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)
    
class UNetDecoder(nn.Module):
    """Unet decoder head"""
    DECODER_TYPE = "unet"

    def __init__(self, in_channels, out_channels, image_size=224, resize_image=False, patch_size=14):
        super(UNetDecoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels
        self.image_size = image_size
        self.resize_image = resize_image
        self.up1 = UNetDecoderUpBlock(in_channels=in_channels, out_channels=in_channels//2, embed_dim=self.embed_dim)
        self.up2 = UNetDecoderUpBlock(in_channels=in_channels//2, out_channels=in_channels//4, embed_dim=self.embed_dim)
        self.up3 = UNetDecoderUpBlock(in_channels=in_channels//4, out_channels=in_channels//8, embed_dim=self.embed_dim)
        self.up4 = UNetDecoderUpBlock(in_channels=in_channels//8, out_channels=out_channels, embed_dim=self.embed_dim)

    def forward(self, x):

        h = w = self.image_size//self.patch_size

        skip1 = x[3].reshape(-1, h, w, self.embed_dim).permute(0,3,1,2)
        skip2 = x[2].reshape(-1, h, w, self.embed_dim).permute(0,3,1,2)
        skip3 = x[1].reshape(-1, h, w, self.embed_dim).permute(0,3,1,2)
        skip4 = x[0].reshape(-1, h, w, self.embed_dim).permute(0,3,1,2)
        x1    = x[4].reshape(-1, h, w, self.embed_dim).permute(0,3,1,2)
        
        x2 = self.up1(x1, skip1)
        x3 = self.up2(x2, skip2)
        x4 = self.up3(x3, skip3)
        x5 = self.up4(x4, skip4)
        
        if self.resize_image:
            x5 = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC)(x5)
        return x5
    
class LinearPostprocessor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, samples, targets):
        logits = self.decoder(samples) 
        if isinstance(logits, list) or (isinstance(logits, torch.Tensor) and len(logits.size()) > 4) : # if 3D output
            logits = torch.cat(logits, dim=0)
            targets = torch.cat(targets, dim=0).cuda()

        preds = logits.argmax(dim=1)
        targets = targets.type(torch.int64)

        return {
            "preds": preds,
            "target": targets,
        }

class AllDecoders(nn.Module):
    def __init__(self, decoders_dict, decoder_type):
        super().__init__()
        self.decoders_dict = nn.ModuleDict()
        self.decoders_dict.update(decoders_dict)
        self.decoder_type = decoder_type

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.decoders_dict.items()}

    def __len__(self):
        return len(self.decoders_dict)

def setup_decoders(embed_dim, learning_rates, num_classes=14, decoder_type="linear", is_3d=False, image_size=224, patch_size=14):
    """
    Sets up the multiple segmentors with different hyperparameters to test out the most optimal one 
    """
    decoders_dict = nn.ModuleDict()
    optim_param_groups = []
    for lr in learning_rates:
        if decoder_type == "linear":
            decoder = LinearDecoder(
                embed_dim, num_classes=num_classes, image_size=image_size, patch_size=patch_size
            )
        elif decoder_type == "unet":
            decoder = UNetDecoder(
                in_channels=embed_dim, out_channels=num_classes, image_size=image_size, resize_image=True, patch_size=patch_size
            )
        if is_3d:
            decoder = Model3DWrapper(decoder, per_slice=True)
        decoder = decoder.cuda()
        decoders_dict[
            f"{decoder_type}:lr={lr:.10f}".replace(".", "_")
        ] = decoder
        optim_param_groups.append({"params": decoder.parameters(), "lr": lr})

    decoders = AllDecoders(decoders_dict, decoder_type)
    if distributed.is_enabled():
        decoders = nn.parallel.DistributedDataParallel(decoders)

    return decoders, optim_param_groups

def save_test_results(feature_model, decoder, dataset, output_dir):
    test_results_path = output_dir + os.sep + "test_results" 
    decoder.resize_image = False
    os.makedirs(test_results_path, exist_ok=True)
    for i, (img, _) in enumerate(dataset):

        img_name = dataset.images[i]
        _, affine_matrix = dataset.get_image_data(i, return_affine_matrix=True)

        img = img.cuda(non_blocking=True) 

        features = feature_model(img.unsqueeze(0))
        output = decoder(features, up_size=512)[0]
        output = output.argmax(dim=1)

        nifti_img = nib.Nifti1Image(output
                                    .cpu()
                                    .numpy()
                                    .astype(np.uint8)
                                    .transpose(1, 2, 0), affine_matrix)    
        file_output_dir = test_results_path + os.sep + img_name + ".gz"

        # Save the NIfTI image
        nib.save(nifti_img, file_output_dir)