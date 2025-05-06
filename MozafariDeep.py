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

        # the ctx dictionary is used to track the values needed by STDP to update the convolutional layers 
        #   with each iteration. It is only used during training and is overwritten with each training
        #   iteration for Conv1, Conv2, and Conv3 and since only one layer is trained at a time it 
        #   is not necessary to track ctx for each layer.
        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}

        # Counters to keep track of when to reset the learning rate for all feature maps.
        #   The default is hardcoded to double the LTP learning rate every 500 spikes
        #   These are utilized for the unsupervised learning on Conv1 and Conv2.
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        # Added padding to the input tensor by adding a 2 element boarder of 0's around each feature (channel)
        #  This is necessary for properly handling the border conditions for the Convolutional window for Conv1
        input = sf.pad(input=input.float(), pad=(2,2,2,2), value=0)

        if self.training:
            # Passing the input tensor to conv1 implicitly calls the __call__() method inherited from nn.Module
            #   It completes the follows steps in order:
            #   1. It sets up the necessary hooks -- the functions we've registered to be executed before the forward pass
            #   2. It executes the forward() method
            #   3. It activiates the backward hooks
            #   4. It returns the output tensor computed by the forward() method, in this case, the neuron potentials
            # Note that is a call to the forward() method inside of snn.Convolution, not a recursive call to this 
            #    forward() method.
            pot = self.conv1(input)

            # Computes the spike-wave tensor from tensor of potentials. Applies a threshold on potentials by 
            #     which all of the values lower or equal to the threshold becomes zero. If threshold is None, 
            #     all the neurons emit one spike (if the potential is greater than zero) in the last time step.
            #  pot - tensor with the thresholded potentials 
            #  spk - tensor with [0,1] depending upon whether the corresponding potential is 0 or positive.
            spk, pot = sf.fire(potentials=pot, threshold=self.conv1_threshold, return_thresholded_potentials=True)

            # From the paper, the first convolutional layer, Conv1, must be trained before the second convolutional
            #    layer, Conv2. I'm not a big fan of having the return statements nested within the conditionals
            #    because it makes the code difficult to follow.
            if max_layer == 1:
                # Adaptive learning rates are mentioned briefly in the SpykeTorch paper, but the implementation is not
                #    discussed. Every 500 iterations, the LTP learning rate (ap) is doubled and then the min operation
                #    is performed against max_ap to place an upper bound on the LTP learning rate.  The LDP learning 
                #    rate (an) is directly calculated from the LTP rate and then these are used to update the 
                #    STDP learning rates for all Conv1 feature maps.
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap=ap.item(), an=an.item())

                # Performs point-wise inhibition between feature maps. After inhibition, at most one neuron is 
                # allowed to fire at each position, which is the neuron with the earliest spike time. If the spike times 
                # are the same, the neuron with the maximum potential will be chosen. As a result, the potential of all of
                # the inhibited neurons will be reset to zero.
                pot = sf.pointwise_inhibition(thresholded_potentials=pot)

                # Returns a tensor (spk) that contains the signs [-1,0,1] of the corresponding potentials from pot. At the 
                #    moment I do not see how these can be -1, but a 1 or a 0 would seem to indicate spike or no spike.
                spk = pot.sign()

                # Finds at most kwta winners first based on the earliest spike time, then based on the maximum potential. It
                # returns a list of winners, each in a tuple of form (feature, row, column).
                winners = sf.get_k_winners(potentials=pot, kwta=self.conv1_kwta, inhibition_radius=self.conv1_inhibition_radius, spikes=spk)

                # Store the CTX values for this iteration for use by the unsupervised STDP training step.
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners

                # Completed current convolutional step, return the thresholded and inhibited potentials tensor as well as
                #    the corresponding spike tensor.
                return spk, pot
            
            # There is a lot packed into the following line. First a max pooling operation is performed on the spike tensor
            #    for each of the features from the first convolutional layer, Conv1. The kernel (window) size is 2x2 with a 
            #    stride of 2. Max pooling essentially downsamples the feature maps by selecting and returning the maximum
            #    value within the observation window, setting the appropriate neuron in the corresponding feature map of the
            #    pooling layer to that max value. With a window of 2x2, the effective number of neurons in the corresponding
            #    feature maps is reduced by a factor of 4.
            # After the pooling operation is completed, the pooled feature maps are padded with a border of zeros, 
            #    one neuron wide.
            spk_in = sf.pad(input=sf.pooling(input=spk, kernel_size=2, stride=2), pad=(1,1,1,1))

            # Execute the second convolutional layer on the spike-wave tensor and return the raw potential values
            #    producted by Conv2.
            pot = self.conv2(spk_in)

            # Apply the firing threshold to all the feature maps in Conv2 to and return the thresholded potentials
            #    as well as the corresponding spike-wave.
            spk, pot = sf.fire(potentials=pot, threshold=self.conv2_threshold, return_thresholded_potentials=True)

            # Similar to above, train the second convolutional layer, Conv2, before training the third convolutional layer, Conv3.
            #    The same adaptive learning rate algorithm as Conv1 is used for training Conv2. 
            # Completed current convolutional step, return the thresholded and inhibited potentials tensor as well as
            #    the corresponding spike tensor.
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
                winners = sf.get_k_winners(potentials=pot, kwta=self.conv2_kwta, inhibition_radius=self.conv2_inhibition_radius, spikes=spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            
            # A max pooling operation is performed on the spike tensor
            #    for each of the features from the second convolutional layer, Conv2. The kernel (window) size is 3x3 with a 
            #    stride of 3. Max pooling essentially downsamples the feature maps by selecting and returning the maximum
            #    value within the observation window, setting the appropriate neuron in the corresponding feature map of the
            #    pooling layer to that max value. With a window of 3x3, the effective number of neurons in the corresponding
            #    feature maps is reduced by a factor of 9.
            # After the pooling operation is completed, the pooled feature maps are padded with a border of zeros, 
            #    two neurons wide.
            spk_in = sf.pad(input=sf.pooling(input=spk, kernel_size=3, stride=3), pad=(2,2,2,2))

            # Execute the third convolutional layer on the spike-wave tensor and return the raw potential values
            #    producted by Conv3.
            pot = self.conv3(spk_in)

            # Compute the spike-wave tensor of potentials. This has different behavior than the previous two layers
            #    because the threshold is set to None. As a result all the neurons emit one spike if the potential 
            #    is greater than zero in the last time step. 
            spk = sf.fire(potentials=pot,threshold=None)

            # Perform the first part of the global max pooling and decision making layer and choose
            #    a single winner across all of the features.
            winners = sf.get_k_winners(potentials=pot, kwta=1, inhibition_radius=0, spikes=spk)

            # Store the CTX values for this iteration for use by the reinforced R-STDP training step.
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners

            # The winners list is used to index into the decision map based upon the winning feature.
            #   The notation winners[0] specifies that we should choose the first winner from the list.
            #   Since there is at most only 1 winner, if there is a winner it is at index 0.
            #   Each winner consists of a tuple of the form (feature, row, column) and would be accessed
            #   as follows:
            #   winner[0][0] - winning feature
            #   winner[0][1] - winning row
            #   winner[0][2] - winning column
            # There are 200 features in Conv3 and these correspond to the 200 elements in the decision_map 
            #    with values in the range of [0,9]. 
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        
        # If we are not training, this process becomes significantly more straight forward.
        else:
            # Execute convolutional layer Conv1 on input spike-wave.
            pot = self.conv1(input)
            
            # Compute thresholded potentials and corresponding spike-wave
            spk, pot = sf.fire(pot, self.conv1_threshold, True)
            
            # I'm not certain why we would want to stop after the first layer
            #    if we are not training, but here we are.
            if max_layer == 1:
                return spk, pot
            
            # Perform pooling on Conv1 spike-wave with a 2x2 kernel and a stride of 2. Pad the results
            #    with a border of zeros 1 neuron wide. The Execute the second convolutional layer, Conv2
            #    on the padded spike-wave, producing tensor of potentials from Conv2.
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))

            # Compute thresholded potentials and corresponding spike-wave
            spk, pot = sf.fire(pot, self.conv2_threshold, True)
            
            # Again, I'm not certain why we would want to stop after the first layer
            #    if we are not training, but here we are.
            if max_layer == 2:
                return spk, pot
            
            # Perform pooling on Conv2 spike-wave with a 3x3 kernel and a stride of 3. Pad the results
            #    with a border of zeros 2 neurons wide. The Execute the second convolutional layer, Conv3
            #    on the padded spike-wave, producing tensor of potentials from Conv3.
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))

            # Compute the spike-wave tensor of potentials. This has different behavior than the previous two layers
            #    because the threshold is set to None. As a result all the neurons emit one spike if the potential 
            #    is greater than zero in the last time step. 
            spk = sf.fire(pot)

            # Perform the first part of the global max pooling and decision making layer and choose
            #    a single winner across all of the features.
            winners = sf.get_k_winners(pot, 1, 0, spk)

            # Use the winning feature to index into the decision map to determine the output for this input image.
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    
    # Apply STDP to the convolutional layer specified by layer_idx, using the values
    #    stored in ctx from the forward() pass.
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    # Update the LTP(ap) and LTD(an) learning rates for the R-STDP operations
    #    R-STDP does not employ the same adaptive learning rates that are implemented 
    #    for the unsupervised STDP in layers 1 and 2. 
    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    # Reward the network by applying STDP to reinforce the strength of the existing weights on Conv3.
    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    # Punish the network by applying ANTI-STDP to reduce the weights on Conv3.
    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

#
# Wrap up the unsupervised training functionality that is used for Conv1 and Conv2
#    network - The instance of the MozafariMNIST2018 network 
#    data - MNIST training dataset with the s1c1 transformation applied
#    layer_idx - index of the layer to train [1,2]
def train_unsupervise(network, data, layer_idx):
    # place the network in training mode
    network.train()
    # process each image in the training dataset
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        # Execute the network on the current input image i to train layer_idx
        network(data_in, layer_idx)
        # Apply STDP to layer_idx
        network.stdp(layer_idx)

#
# Wrap up the R-STDP training functionality that is used for Conv2
#    network - The instance of the MozafariMNIST2018 network 
#    data - MNIST training dataset with the s1c1 transformation applied
#    target - these are the target values we expect the network to predict
#               for the corresponding values in the training data.
# Return a np.array showing the fraction correct, wrong, and silent
def train_rl(network, data, target):
    # place the network in training mode
    network.train()
    # declare an array to track the training performance
    perf = np.array([0,0,0]) # correct, wrong, silence
    # process each image in the training dataset
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        # Execute the network on the current input image i for Conv3 and capture the
        #    output value derived from the decision map. A -1 indicates that
        #    no winning feature maps were found for Conv3 for the current image.
        d = network(data_in, 3)
        if d != -1:
            # output == target -> reward()
            if d == target_in:
                perf[0]+=1
                network.reward()
            # output != target -> punish()
            else:
                perf[1]+=1
                network.punish()
        # No winning feature map identified. No updates to Conv3 using STDP aka silence
        else:
            perf[2]+=1
    return perf/len(data)

#
# Network Test. This is identical to train_rl except it doesn't not make calls to 
#     reward or punish the network using STDP.
#
#    network - The instance of the MozafariMNIST2018 network 
#    data - MNIST training dataset with the s1c1 transformation applied
#    target - these are the target values we expect the network to predict
#               for the corresponding values in the training data.
# Return a np.array showing the fraction correct, wrong, and silent
def test(network, data, target):
    # place the network in evaluation mode
    network.eval()
    # declare an array to track the training performance
    perf = np.array([0,0,0]) # correct, wrong, silence
    # process each image in the training dataset
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        # Execute the network on the current input image i for Conv3 and capture the
        #    output value derived from the decision map. A -1 indicates that
        #    no winning feature maps were found for Conv3 for the current image.
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
