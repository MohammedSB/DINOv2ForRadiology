import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn

import dinov2.distributed as distributed
from dinov2.eval.utils import is_zero_matrix

class DINOV2Encoder(torch.nn.Module):
    def __init__(self, encoder, autocast_ctx, is_3d=False) -> None:
        super(DINOV2Encoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.autocast_ctx = autocast_ctx
        self.is_3d = is_3d
    
    def forward_3d(self, x):
        batch_features = [] 
        for batch_scans in x: # calculate the features for every scan in all scans of the batch
            scans = []
            for scan in batch_scans:
                if not is_zero_matrix(scan): scans.append(self.forward_(scan.unsqueeze(0)))
            batch_features.append(scans)
        return batch_features

    def forward_(self, x):
        with torch.no_grad():
            with self.autocast_ctx():
                features = self.encoder.forward_features(x)['x_norm_patchtokens']
        return features

    def forward(self, x):
        if self.is_3d:
            return self.forward_3d(x)
        return self.forward_(x)

class LinearDecoder(torch.nn.Module):
    """Linear decoder head"""
    DECODER_TYPE = "linear"

    def __init__(self, in_channels, tokenW=32, tokenH=32, num_classes=3, is_3d=False):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.decoder = torch.nn.Conv2d(in_channels, num_classes, (1,1))
        self.decoder.weight.data.normal_(mean=0.0, std=0.01)
        self.decoder.bias.data.zero_()
        self.is_3d = is_3d

    def forward_3d(self, embeddings, up_size, vectorized=False):
        batch_outputs = []
        for batch_embeddings in embeddings:
            if vectorized:
                batch_outputs.append(self.forward_(torch.stack(batch_embeddings, up_size).squeeze()))
            else:
                batch_outputs.append(
                    torch.stack([self.forward_(slice_embedding, up_size) for slice_embedding in batch_embeddings]).squeeze()
                    )
        return batch_outputs

    def forward_(self, embeddings, up_size):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        output = self.decoder(embeddings)
        
        # Upsample (interpolate) output/logit map.
        output = torch.nn.functional.interpolate(output, size=up_size, mode="bilinear", align_corners=False)

        return output
    
    def forward(self, embeddings, up_size=448):
        if self.is_3d:
            return self.forward_3d(embeddings, up_size)
        return self.forward_(embeddings, up_size)
    
class LinearPostprocessor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, samples, targets):
        logits = self.decoder(samples) 
        if isinstance(logits, list): # if 3D output
            logits = torch.cat(logits, dim=0)
            targets = torch.cat(targets, dim=0).cuda()

        preds = logits.argmax(dim=1)
        targets = targets.type(torch.int64)

        return {
            "preds": preds,
            "target": targets,
        }

class AllDecoders(nn.Module):
    def __init__(self, decoders_dict):
        super().__init__()
        self.decoders_dict = nn.ModuleDict()
        self.decoders_dict.update(decoders_dict)
        self.decoder_type = list(decoders_dict.values())[0].DECODER_TYPE

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.decoders_dict.items()}

    def __len__(self):
        return len(self.decoders_dict)

def setup_decoders(embed_dim, learning_rates, num_classes=14, decoder_type="linear", is_3d=False):
    """
    Sets up the multiple segmentors with different hyperparameters to test out the most optimal one 
    """
    decoders_dict = nn.ModuleDict()
    optim_param_groups = []
    for lr in learning_rates:
        if decoder_type == "linear":
            decoder = LinearDecoder(
                embed_dim, num_classes=num_classes, is_3d=is_3d
            )
        decoder = decoder.cuda()
        decoders_dict[
            f"{decoder_type}:lr={lr:.10f}".replace(".", "_")
        ] = decoder
        optim_param_groups.append({"params": decoder.parameters(), "lr": lr})

    decoders = AllDecoders(decoders_dict)
    if distributed.is_enabled():
        decoders = nn.parallel.DistributedDataParallel(decoders)

    return decoders, optim_param_groups

def save_test_results(feature_model, decoder, dataset, output_dir):
    test_results_path = output_dir + os.sep + "test_results" 
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