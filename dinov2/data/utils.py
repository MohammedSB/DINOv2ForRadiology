import pandas as pd
import torch

def get_fewshot_in_nih(d, shots=8):
    if isinstance(d, torch.utils.data.ConcatDataset):
        df = pd.concat([d.datasets[0].labels, d.datasets[1].labels], axis=0).reset_index(drop=True)
        class_names = d.datasets[0].class_names
    else:
        df = d.labels
        class_names = d.class_names
    
    indices = []
    df['Patient ID'] = df['Image Index'].apply(lambda x: x.split('_')[0])

    grouped = df.groupby('Patient ID')['Finding Labels']
    df_new = grouped.apply(lambda x: list(set.intersection(*map(set, x))))  # Get intersection of all label sets

    added_patients = []
    for class_ in class_names:
        patient_ids = df_new[df_new.apply(lambda x: class_ in x)].index.tolist()
        per_class_indices = []
        for patient_id in patient_ids[:shots]:
            if patient_id in added_patients:
                continue
            added_patients.append(patient_id)
            per_class_indices.extend(df[df['Patient ID'] == patient_id].index.tolist())
        indices.extend(per_class_indices)
    return indices