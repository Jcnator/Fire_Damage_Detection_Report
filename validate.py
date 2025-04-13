from models.model_loader import ClassifierModel
from dataloader.dataloader import get_dataloader
from eval_metrics import get_accuracy, get_mAP
import torch
import wandb
from tqdm import tqdm
from options import BaseOptions

import os
import argparse

from utils import save_image_preds

            
def validation_step(classifier, optim, val_loader, epoch, args):
    classifier.eval()

    data_loader = iter(val_loader)
    num_iters = len(val_loader)
    class_correct = torch.zeros((3), dtype=torch.int32)
    class_total = torch.zeros((3), dtype=torch.int32)
    total_loss = 0
    correct = 0
    total = 0
    class_tp = torch.zeros((3), dtype=torch.float32)
    class_fp = torch.zeros((3), dtype=torch.float32)
    #for i, data in enumerate(trainloader):
    # pbar = tqdm(range(num_iters))

    with torch.no_grad():
        classifier.eval()
        pbar = tqdm(range(num_iters))
        for i in pbar:
            data = next(data_loader)
            im_data, ann_data = data
            pred = classifier(im_data)
            loss = classifier.loss(pred, ann_data)

            if(args.save_images):
                # Test Mode !!!
                save_image_preds(im_data, pred, ann_data)

            total_loss += loss.item()
            #accuracy
            curr_correct, curr_total, curr_class_correct, curr_class_total = get_accuracy(pred, ann_data)
            class_correct += curr_class_correct
            class_total += curr_class_total
            curr_tp, curr_fp = get_mAP(pred, ann_data)
            class_tp += curr_tp
            class_fp += curr_fp
            correct += curr_correct
            total += curr_total

            
            pbar.set_description('Validation  Epoch: {}, Loss: {:.4f}, Val Acc: {:.4f}, Val mAP:{:.4f}'.format(epoch, total_loss / (i+1), correct/total, torch.mean(class_tp / (class_tp + class_fp + 1e-6))))
            
    val_loss = total_loss / (i+1)
    val_acc = correct/total
    val_class_AP = class_tp / (class_tp + class_fp + 1e-6)
    print("Class AP", val_class_AP)
    val_mAP = val_class_AP.sum() / 3

    class_accuracy = [float(f"{x:.4f}") for x in class_correct.float() / (class_total.float()+1e-6)]
    print(f"Class Correct: {class_correct.tolist()}")
    print(f"Class Total: {class_total.tolist()}")
    print(f"Class Accuracy: {class_accuracy}")
  
    return val_loss, val_acc, val_mAP


if __name__ == '__main__':
    opts = BaseOptions()
    parser = argparse.ArgumentParser()
    #HyperParams
    parser.add_argument("--arch", type=str, default=opts.arch)
    parser.add_argument("--batch_size", type=int, default=opts.batch_size)
    # Saving and Loggin
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_model_epoch", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default=opts.save_dir)
    parser.add_argument("--saved_model_path", type=str, default="results/ResNet18_rgb_NIR_spectra_and_blur.pth")
    parser.add_argument("--validation_set", type=str, default="validate")
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--no-save_images', dest='save_images', action='store_false')
    parser.set_defaults(save_images=True)


    args = parser.parse_args()
    args = opts.add_aditional_opts(args)

    if args.validation_set == 'validation':
        data_loader = get_dataloader(args, "validate", augment=False)
    else:
        data_loader = get_dataloader(args, "test", augment=False)

    classifier = ClassifierModel(opts.arch, opts).to(opts.device)
    optim = torch.optim.Adam(classifier.parameters(), lr=opts.lr, betas=(0.5, 0.999))
    classifier.load_state_dict(torch.load(args.saved_model_path, weights_only=True))
    validation_step(classifier,optim,data_loader, 99, args)