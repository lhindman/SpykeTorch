from SpykeTorch import utils
from torchvision import transforms
from MozafariDeep import *

# These settings define the 6 DoG Kernels described in the paper that are used to
#    generate the six feature maps in the Intensity to Latency encoding layer.
kernels = [ utils.DoGKernel(window_size=3,sigma1=3/9,sigma2=6/9),
            utils.DoGKernel(window_size=3,sigma1=6/9,sigma2=3/9),
            utils.DoGKernel(window_size=7,sigma1=7/9,sigma2=14/9),
            utils.DoGKernel(window_size=7,sigma1=14/9,sigma2=7/9),
            utils.DoGKernel(window_size=13,sigma1=13/9,sigma2=26/9),
            utils.DoGKernel(window_size=13,sigma1=26/9,sigma2=13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)

s1c1 = S1C1Transform(filter)

#
# Load the MNIST datasets, apply the s1c1 transformation filter as the data sets are loaded and 
#    leverage the CacheDataset wrapper to improve performance.
#
data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

#
# Initialize the network
#
mozafari = MozafariMNIST2018()

#
# Setup CUDA
#
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    if use_cuda:
        print("Activating CUDA on network")
        mozafari.cuda()

else:
    print("CUDA is not available")


#
# Training The First Layer
#
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
#
# Training The Second Layer
#
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

#
# initial adaptive learning rates
# Unfortunately, I cannot find the documentation on how these 
#     adaptive learning rates work with R-STDP in Conv3.
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

#
# Training The Third Layer and test
#
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
