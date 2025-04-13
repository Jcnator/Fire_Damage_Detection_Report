import torch

def get_accuracy(pred, gt):
    total = gt.shape[0]
    num_classes = gt.shape[1]
    pred = torch.argmax(pred, dim=1)
    gt = torch.argmax(gt, dim=1)
    correct = torch.sum(pred == gt).sum().item()
    class_correct = torch.zeros(num_classes, dtype=torch.int32)
    class_total = torch.zeros(num_classes, dtype=torch.int32)
    for i in range(num_classes):
        class_total[i] = torch.sum(gt==i).item()
        class_correct[i] = torch.sum((pred == i) & (gt == i)).item()
    return correct , total, class_correct, class_total

def get_precision_and_recall(pred, gt):
    num_class = pred.shape[1]
    
    pred = torch.argmax(pred, dim=1)
    gt = torch.argmax(gt, dim=1)
    class_tp = torch.zeros((num_class))
    class_fp = torch.zeros((num_class))
    class_fn = torch.zeros((num_class))
    for curr_pred, curr_gt in zip(pred, gt):
            if curr_pred == curr_gt:
                class_tp[curr_gt] += 1
            else:
                class_fp[curr_pred] += 1

def get_mAP(pred, gt):
    num_class = pred.shape[1]
    #print(num_class)
    class_tp = torch.zeros((num_class))
    class_fp = torch.zeros((num_class))
    pred = torch.argmax(pred, dim=1)
    gt = torch.argmax(gt, dim=1)
    #print(pred)
    #print(gt)
    
    for curr_pred, curr_gt in zip(pred, gt):
        if curr_pred == curr_gt:
            class_tp[curr_gt] += 1
        else:
            class_fp[curr_pred] += 1
    #print(class_precision)
    return class_tp, class_fp