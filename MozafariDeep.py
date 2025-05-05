#################################################################################
# Reimplementation of the 10-Class Digit Recognition Experiment Performed in:   #
# https://arxiv.org/abs/1804.00227                                              #
#                                                                               #
# Reference:                                                                    #
# Mozafari, Milad, et al.,                                                      #
# "Combining STDP and Reward-Modulated STDP in                                  #
# Deep Convolutional Spiking Neural Networks for Digit Recognition."            #
# arXiv preprint arXiv:1804.00227 (2018).                                       #
#                                                                               #
# Original implementation (in C++/CUDA) is available upon request.              #
#################################################################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda = True

class MozafariMNIST2018(nn.Module):
    def __init__(self):
        # Call the init function for nn.Module
        super(MozafariMNIST2018, self).__init__()

        # Setup Convolutional Layer 1 with 6 input channels
        #     30 output channels (features) and a kernel size of 5x5 with a stride of 1
        #   Random weights are assigned to each kernel using a normal distribution
        #     with an average of 0.8 and a standard deviation of 0.05.
        #
        # The firing threshold for neurons in Conv1 is set to be 15
        # Number of winners (kwta) for Conv1 is set to 5
        # The inhibition radius for Conv1 is set to 3.
        self.conv1 = snn.Convolution(in_channels=6, out_channels=30, kernel_size=5, weight_mean=0.8, weight_std=0.05)
        self.conv1_threshold = 15
        self.conv1_kwta = 5
        self.conv1_inhibition_radius = 3

        # Setup Convolutional Layer 2 with 30 input channels, 
        #     250 output channels (features) and a kernel size of 3x3 with a stride of 1.
        #  Random weights are assigned to each kernel using a normal distriution
        #     with an average of 0.8 and a standard deviation of 0.05. This is the same as Conv1 above.
        #
        #  The firing threshold for neurons in Conv2 is set to be 10
        #  Number of winnders (kwta) for Conv2 is 8
        #  The inhibition radius for Conv1 is 1
        self.conv2 = snn.Convolution(in_channels=30, out_channels=250, kernel_size=3, weight_mean=0.8, weight_std=0.05)
        self.conv2_threshold = 10
        self.conv2_kwta = 8
        self.conv2_inhibition_radius = 1

        # Setup Convolutional Layer 3 with 250 input channels,
        #     200 output channels (features) and a kernel size of 5x5 with a stride of 1.
        #  Random weights are assigned to each kernel using a normal distriution
        #     with an average of 0.8 and a standard deviation of 0.05. This is the same as Conv1 and Conv2 above. 
        self.conv3 = snn.Convolution(in_channels=250, out_channels=200, kernel_size=5, weight_mean=0.8, weight_std=0.05)

        # Define STDP learning rates to use for unsupervised learning on Conv1 and Conv2
        self.stdp1 = snn.STDP(conv_layer=self.conv1, learning_rate=(0.004, -0.003))
        self.stdp2 = snn.STDP(conv_layer=self.conv2, learning_rate=(0.004, -0.003))

        # Configure STDP and ANTI-STDP to emulate the behavior of R-STDP on Conv3
        self.stdp3 = snn.STDP(conv_layer=self.conv3, learning_rate=(0.004, -0.003), use_stabilizer=False, lower_bound=0.2, upper_bound=0.8)
        self.anti_stdp3 = snn.STDP(conv_layer=self.conv3, learning_rate=(-0.004, 0.0005), use_stabilizer=False, lower_bound=0.2, upper_bound=0.8)
        
        # Specify the max LTP learning rate of 0.15
        #    The learning rate is doubled every 500 spikes and this
        #     values sets the upper limit.
        self.max_ap = Parameter(torch.Tensor([0.15]))

        # Initialize the decision map list with the following:
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        # The values in the decision map are used to represent the different classifier 
        # outputs in the range of [1,9]. There are 20 instances of each of the 10 possible 
        # outputs for a total of 200 mappings that correspond to the 200 out_channels from Conv3. 
        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i]*20)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}

        # Counters to keep track of when to reset the learning rate for all feature maps.
        #   The default is hardcoded to double the LTP learning rate every 500 spikes
        #   These are utilized for the unsupervised learning on Conv1 and Conv2.
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(potentials=pot, threshold=self.conv1_threshold, return_thresholded_potentials=True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(potentials=pot, kwta=self.conv1_kwta, inhibition_radius=self.conv1_inhibition_radius, spikes=spk)
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_threshold, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.conv2_kwta, self.conv2_inhibition_radius, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_threshold, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_threshold, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def train_rl(network, data, target):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

def test(network, data, target):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)

class S1C1Transform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()

kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1C1Transform(filter)

data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

mozafari = MozafariMNIST2018()

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

else:
    print("CUDA is not available")

if use_cuda:
    mozafari.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
    mozafari.load_state_dict(torch.load("saved_l1.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 1)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), "saved_l1.net")
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
    mozafari.load_state_dict(torch.load("saved_l2.net"))
else:
    for epoch in range(4):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 2)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), "saved_l2.net")

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

# Training The Third Layer
print("Training the third layer")
for epoch in range(680):
    print("Epoch #:", epoch)
    perf_train = np.array([0.0,0.0,0.0])
    for data,targets in MNIST_loader:
        perf_train_batch = train_rl(mozafari, data, targets)
        print(perf_train_batch)
        #update adaptive learning rates
        apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
        perf_train += perf_train_batch
    perf_train /= len(MNIST_loader)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)

    for data,targets in MNIST_testLoader:
        perf_test = test(mozafari, data, targets)
        if best_test[0] <= perf_test[0]:
            best_test = np.append(perf_test, epoch)
            torch.save(mozafari.state_dict(), "saved.net")
        print(" Current Test:", perf_test)
        print("    Best Test:", best_test)
