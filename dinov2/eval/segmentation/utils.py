import torch
import torch.nn as nn

import dinov2.distributed as distributed

class TransformerEncoder(torch.nn.Module):
    def __init__(self, encoder, autocast_ctx) -> None:
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.autocast_ctx = autocast_ctx
    
    def forward(self, x):
        with torch.no_grad():
            with self.autocast_ctx():
                features = self.encoder.forward_features(x)['x_norm_patchtokens']
        return features

class LinearDecoder(torch.nn.Module):
    """Linear decoder head"""
    DECODER_TYPE = "linear"

    def __init__(self, in_channels, tokenW=32, tokenH=32, num_classes=3):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.decoder = torch.nn.Conv2d(in_channels, num_classes, (1,1))
        self.decoder.weight.data.normal_(mean=0.0, std=0.01)
        self.decoder.bias.data.zero_()

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.decoder(embeddings)
    
class LinearPostprocessor(nn.Module):
    def __init__(self, decoder, class_mapping=None):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        logits = self.decoder(samples)
        logits = torch.nn.functional.interpolate(logits, size=targets.shape[2], mode="bilinear", align_corners=False)
        
        preds = logits.argmax(dim=1)
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

def setup_decoders(embed_dim, learning_rates, num_classes=14, decoder_type="linear"):
    """
    Sets up the multiple segmentors with different hyperparameters to test out the most optimal one 
    """
    decoders_dict = nn.ModuleDict()
    optim_param_groups = []
    for lr in learning_rates:
        if decoder_type == "linear":
            decoder = LinearDecoder(
                embed_dim, num_classes=num_classes
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