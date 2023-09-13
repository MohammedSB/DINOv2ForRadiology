import torch
import torch.nn as nn

from dinov2.eval.utils import is_zero_matrix

def classifier_forward_pass(feature_model, classifier_heads, data, is_3d=False):
    if is_3d:
        batch_features = [] 
        for batch_scans in data:
            scans = []
            for scan in batch_scans:
                if not is_zero_matrix(scan): scans.append(feature_model(scan.unsqueeze(0)))
            batch_features.append(scans)
        outputs = [
            list(classifier_heads(batch_feature).values()) for batch_feature in batch_features
            ]
        classifier_outputs = []
        for output in outputs:
            classifier_outputs.append(torch.stack(output).squeeze())
        outputs = torch.stack(classifier_outputs, dim=1)
        outputs = {
            list(classifier_heads.module.classifiers_dict.keys())[i]: output
            for i, output in enumerate(outputs)
            }
    else:
        features = [feature_model(data)]
        outputs = classifier_heads(features)

    return outputs