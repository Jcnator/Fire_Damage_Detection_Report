import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

random.seed(69)
def save_image_preds(im_data, pred, ann_data, save_root='image_predictions'):
    # Denormalize - vibe coded from gpt.
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5], device=im_data.device)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5], device=im_data.device)
    im_data = im_data * std[None, :, None, None] + mean[None, :, None, None]
    
    pred_labels = torch.argmax(pred, dim=1)
    true_labels = torch.argmax(ann_data, dim=1)

    for idx in range(im_data.shape[0]):
        gt = true_labels[idx].item()
        pred_label = pred_labels[idx].item()

        rgb = im_data[idx][:3] 
        rgb = torch.clamp(rgb, 0, 1).cpu()

        fig, ax = plt.subplots()
        ax.imshow(TF.to_pil_image(rgb))
        ax.axis('off')
        ax.set_title(f'GT: {gt}, Pred: {pred_label}', fontsize=12)

        save_dir = os.path.join(save_root, f'{gt}_{pred_label}')
        os.makedirs(save_dir, exist_ok=True)
        suffix = random.randint(0,999)
        save_path = os.path.join(save_dir, f'image_{idx}_{suffix}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
