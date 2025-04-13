import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

NIR_MIN = 780
NIR_MAX = 1400
NIR_RANGE = NIR_MAX - NIR_MIN

save_path = "dataloader/"

'''
Preprocess the NIR for each data class to get the gaussian spectra
https://www.sciencedirect.com/science/article/pii/S0169743923001740
'''
def save_nir_gaussian_values(dataloader):
    print("save_nir_gaussian_values")
    class_0_data = np.zeros((256,227,227))
    class_1_data = np.zeros((46,227,227))
    class_2_data = np.zeros((794,227,227))

    data_loader = iter(dataloader)
    num_iters = len(dataloader)
    pbar = tqdm(range(num_iters))
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for i in pbar:
        im_data, ann_data = next(data_loader)
        nir_data = im_data.detach().cpu().numpy()[:,3,:,:] 
        #print(im_data.shape)
        ann_data = ann_data.detach().cpu().numpy()
        class_idx = np.argmax(ann_data)
        if class_idx == 0:
            class_0_data[count_0] = nir_data 
            count_0 += 1
        elif class_idx == 1:
            class_1_data[count_1] = nir_data
            count_1 += 1

        else:
            class_2_data[count_2] = nir_data
            count_2 += 1

    
    print("Class Data")
    print(len(class_0_data))
    print(len(class_1_data))
    print(len(class_2_data))
    total_min, total_max = get_total_min_and_max([class_0_data, class_1_data, class_2_data])
    print("Min Max")
    print(total_min, total_max)

    class_0_samples = get_sample_matrix(class_0_data, total_min, total_max)
    class_1_samples = get_sample_matrix(class_1_data, total_min, total_max)
    class_2_samples = get_sample_matrix(class_2_data, total_min, total_max)
    # plt.plot(np.average(class_0_samples,axis=0))
    # plt.show()


    # plt.plot(np.average(class_1_samples,axis=0))
    # plt.show()

    # plt.plot(np.average(class_2_samples,axis=0))
    # plt.show()


    print(np.min(class_0_samples), np.max(class_0_samples))
    print(np.min(class_1_samples), np.max(class_1_samples))
    print(np.min(class_2_samples), np.max(class_2_samples))

    class_0_cov = get_covar(class_0_samples)
    class_1_cov = get_covar(class_1_samples)
    class_2_cov = get_covar(class_2_samples)


    # class_0_cov = np.cov(class_0_samples)
    # class_1_cov = np.cov(class_1_samples)
    # class_2_cov = np.cov(class_2_samples)
    print(np.min(class_0_cov), np.max(class_0_cov))
    print(np.min(class_1_cov), np.max(class_1_cov))
    print(np.min(class_2_cov), np.max(class_2_cov))
    np.save(os.path.join(save_path,"class_0_cov.npy"), class_0_cov)
    np.save(os.path.join(save_path,"class_1_cov.npy"), class_1_cov)
    np.save(os.path.join(save_path,"class_2_cov.npy"), class_2_cov)


def get_covars_from_data(data, annotations, opts):
    print(len(data))
    class_count = opts.class_count
    class_0_data = np.zeros((int(class_count[0]*opts.test_train_split[0]),opts.image_size,opts.image_size))
    class_1_data = np.zeros((int(class_count[1]*opts.test_train_split[0]),opts.image_size,opts.image_size))
    class_2_data = np.zeros((int(class_count[2]*opts.test_train_split[0]),opts.image_size,opts.image_size))
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for im, ann_data in zip(data, annotations):
        im_data = torch.tensor(im)
        nir_data = im_data[3,:,:] 
        class_idx = np.argmax(ann_data)
        #print(class_idx)
        if class_idx == 0:
            class_0_data[count_0] = nir_data 
            count_0 += 1
        elif class_idx == 1:
            class_1_data[count_1] = nir_data
            count_1 += 1

        else:
            class_2_data[count_2] = nir_data
            count_2 += 1

    total_min, total_max = get_total_min_and_max([class_0_data, class_1_data, class_2_data])
    print("Total Min Max",total_min, total_max )
    class_0_samples = get_sample_matrix(class_0_data, total_min, total_max)
    class_1_samples = get_sample_matrix(class_1_data, total_min, total_max)
    class_2_samples = get_sample_matrix(class_2_data, total_min, total_max)

    class_0_cov = get_covar(class_0_samples)
    class_1_cov = get_covar(class_1_samples)
    class_2_cov = get_covar(class_2_samples)

    return class_0_cov, class_1_cov, class_2_cov


# We want to trun the NxK (k is num pixesl) NIR data into NxP where P is the wavelength sampling
# 
def get_sample_matrix(class_data, total_min, total_max, num_samples = 12):
    N = len(class_data)
    class_min = np.min(class_data)
    class_max = np.max(class_data)
    data_samples = np.zeros((N,num_samples))
    for i, x in enumerate(class_data):
        #print(x.shape)
        sample_x, _ = np.histogram(x, num_samples,range=(total_min,total_max))
        #print(sample_x.shape)
        data_samples[i] = normalize(np.array(sample_x))
    return data_samples

def normalize(sample_x):
        pc_min = np.min(sample_x)
        pc_max = np.max(sample_x) 
        #print(point_cloud.shape, pc_min.shape, pc_max.shape)
        sample_x = (sample_x - pc_min) / (pc_max - pc_min)
        sample_x = 2*sample_x-1
        ##print(torch.min(point_cloud), torch.max(point_cloud))
        #print(sample_x)

        return sample_x

# [N, P] matrix
def get_covar(sample_matrix):
    N, P = sample_matrix.shape
    print("Get Covar")
    print("Sample mat", sample_matrix.shape)
    sample_mean = np.mean(sample_matrix, axis=0) # [1, P] vector
    print("Sample Mean", sample_mean.shape)
    print("Sample Mean", np.mean(sample_mean))
    #sample_mean = np.mean(sample_matrix, axis=1)       # [N, 1] vector
    #print(sample_mean.shape)
    epsilon = 0

    sample_covar = np.zeros((P,P))
    for j in range(P):
        for k in range(P):
            if j == k:
                sample_covar[j,k] = np.sum((sample_matrix[:,j] - sample_mean[j])**2) + epsilon
            else:
                sample_covar[j,k] = np.sum((sample_matrix[:,j] - sample_mean[j])*(sample_matrix[:,k] - sample_mean[k])) 

    sample_covar = sample_covar / (N-1)
    print(sample_covar)
    return sample_covar



def get_total_min_and_max(data):
    total_min = float('inf')
    total_max = float('-inf')
    for entry in data:
        #print(entry)
        #print(np.min(entry))
        total_min = min(np.min(entry), total_min)
        total_max = max(np.max(entry), total_max)
    return total_min, total_max

    

