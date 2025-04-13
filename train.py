from models.model_loader import ClassifierModel
from dataloader.dataloader import get_dataloader
from eval_metrics import get_accuracy, get_mAP
from validate import validation_step
import torch
import wandb
from tqdm import tqdm
from options import BaseOptions

import os
import argparse

class Trainer:
    def __init__(self, opts):
        self.opts = opts 
        self.num_epochs = opts.num_epochs

        self.ckptdir =  os.path.join(self.opts.save_dir, self.opts.arch,  'checkpoints')
        print(self.ckptdir)

        self.classifier = ClassifierModel(opts.arch, opts).to(self.opts.device)
        self.trainloader = get_dataloader(opts, 'train', shuffle=True, augment=True, sample=False)
        self.val_loader = get_dataloader(opts, "validate")
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=20, gamma=0.5)

        if self.opts.logging:
            run = wandb.init(
                # Set the project where this run will be logged
                project=self.opts.project_name + self.opts.arch,
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.opts.lr,
                    "epochs": self.num_epochs,
                    "batch_size": self.opts.batch_size,
                    "augment": self.opts.augment,
                    "standard_augment": self.opts.standard_augment,
                    "nir_augment": self.opts.gaussian_nir,
                    "nir_buckets": self.opts.NIR_buckets,
                },
            )
        


    def training_loop(self):
        for epoch in range(self.num_epochs):
            self.training_step(epoch)
            self.scheduler.step()
            self.val_loss, self.val_acc, self.val_mAP = validation_step(self.classifier, self.optim, self.val_loader, epoch, self.opts)
            if epoch % self.opts.save_freq == 0:
                self.save_step(epoch)
            if self.opts.logging:
                self.log_step(epoch)
        self.save_step(epoch)
            
    def training_step(self, epoch):
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        class_tp = torch.zeros((3), dtype=torch.float32)
        class_fp = torch.zeros((3), dtype=torch.float32)
        pbar = tqdm(self.trainloader)
        for i, data in enumerate(pbar):
            self.optim.zero_grad()
            im_data, ann_data = data
            pred = self.classifier(im_data)
            loss = self.classifier.loss(pred, ann_data)

            curr_correct, curr_total, curr_class_corr, curr_class_total = get_accuracy(pred, ann_data) 
            curr_tp, curr_fp = get_mAP(pred, ann_data)
            class_tp += curr_tp
            class_fp += curr_fp
            correct += curr_correct
            total += curr_total
            accuracy = correct / total


            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            pbar.set_description('Training   Epoch: {}, Loss: {:.4f}, Train Acc: {:.4f}, Train mAP: {:.4f}'.format(epoch, loss.item(), accuracy, torch.mean(class_tp / (class_tp + class_fp + 1e-6))))

            
            
        self.train_loss = total_loss / (i+1)
        self.train_acc = correct / total
        self.train_mAP = torch.mean(class_tp / (class_tp + class_fp + 1e-6))


    def save_step(self, epoch):
        file_name = self.opts.arch +"_" + str(epoch)+".pth"
        os.makedirs(self.ckptdir, exist_ok=True)
        path = os.path.join(self.ckptdir,file_name)
        print(path)
        torch.save(self.classifier.state_dict(), path)

        

    def log_step(self, epoch):
          wandb.log({"train_loss": self.train_loss, 
                    "val_loss": self.val_loss,
                    "train_accuracy": self.train_acc,
                    "val_accuracy": self.val_acc,
                    "train_mAP": self.train_mAP,
                    "val_mAP": self.val_mAP
                    })


if __name__ == '__main__':
    opts = BaseOptions()
    parser = argparse.ArgumentParser()
    #HyperParams
    parser.add_argument("--arch", type=str, default=opts.arch)
    parser.add_argument("--batch_size", type=int, default=opts.batch_size)
    parser.add_argument("--test_train_split", type=float, default=opts.test_train_split)
    parser.add_argument("--lr", type=float, default=opts.lr)
    parser.add_argument("--num_epochs", type=int, default=opts.num_epochs, help="number of training steps")
    parser.add_argument("--dropout", type=float, default=opts.dropout, help="dropout rate")

    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--no-weighted_loss', dest='weighted_loss', action='store_false')
    parser.set_defaults(weighted_loss=opts.class_weights)
    parser.add_argument("--device", type=str, default=opts.device)

    #Training Options
    # Augment
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--augment', dest='augment', action='store_true')
    feature_parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(feature=opts.augment)  
    # Augment RGB
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--augment_RGB', dest='augment_RGB', action='store_true')
    feature_parser.add_argument('--no-augment_RGB', dest='augment_RGB', action='store_false')
    parser.set_defaults(feature=opts.standard_augment)
    # Split RGB and NIR
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--split_RGB_and_NIR', dest='split_RGB_and_NIR', action='store_true')
    feature_parser.add_argument('--no-split_RGB_and_NIR', dest='split_RGB_and_NIR', action='store_false')
    parser.set_defaults(feature=opts.split_rgb_and_nir)


    # Augment NIR
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--augment_NIR', dest='augment_NIR', action='store_true')
    feature_parser.add_argument('--no-augment_NIR', dest='augment_NIR', action='store_false')
    parser.set_defaults(feature=opts.gaussian_nir)   
    parser.add_argument("--NIR_buckets", type=int, default=opts.NIR_buckets, help="number of NIR buckets")

    # Wandb
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--logging', dest='logging', action='store_true',  help="log to wandb?")
    feature_parser.add_argument('--no-logging', dest='logging', action='store_false',  help="don't load to wandb")
    parser.set_defaults(feature=opts.logging)  
    
    # Saving and Loggin
    parser.add_argument("--log_interval", type=int, default=opts.save_freq)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_model_epoch", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default=opts.save_dir)
    parser.add_argument("--saved_model_path", type=str, default="results/load_test/last.ckpt")

    args = parser.parse_args()
    print(args)
    args = opts.add_aditional_opts(args)
   

    trainer = Trainer(args)
    trainer.training_loop()