import numpy as np
import torch
import torch.nn as nn

from dinov2.eval.utils import is_padded_matrix, Model3DWrapper
import dinov2.distributed as distributed

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000, is_3d=False):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        output = torch.stack( # If 3D, take average of all slices.
            [create_linear_input(image, self.use_n_blocks, self.use_avgpool) for image in x]
            ).mean(dim=0).squeeze()
        return self.linear(output).squeeze()

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier):
        super().__init__()
        self.linear_classifier = linear_classifier

    def forward(self, samples, targets):
        preds = torch.sigmoid(self.linear_classifier(samples))
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets).cuda() 
        return {
            "preds": preds,
            "target": targets,
        }

def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, avgpools=[True, False], num_classes=14, is_3d=False):
    """
    Sets up the multiple linear classifiers with different hyperparameters to test out the most optimal one 
    """
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in avgpools:
            for _lr in learning_rates:
                # lr = scale_lr(_lr, batch_size)
                lr = _lr
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                if is_3d:
                    linear_classifier = Model3DWrapper(linear_classifier)
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"linear:blocks={n}:avgpool={avgpool}:lr={lr:.10f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups