# These notes document the MozafariDeep SNN Training Performance on a variety of systems.

## Tesseract
### Configuration
Tesseract is a dual, 8 core Xeon E5-2665 @ 2.4GHz with 256GB RAM, 512GB NVME storage and 2 x 500GB SATA disks in RAID0 configuration. Tesseract has a 10GigE connection to the primary to a 48TB NAS system.


### Simulation Accuracy

```
Current Train: [0.98001667 0.01998333 0.        ]
   Best Train: [9.80016667e-01 1.99833333e-02 0.00000000e+00 6.79000000e+02]
 Current Test: [0.9588 0.0412 0.    ]
    Best Test: [9.607e-01 3.930e-02 0.000e+00 6.640e+02]
```


### Simuation Time
        Command being timed: "python MozafariDeepDriver.py"
        User time (seconds): 284015.10
        System time (seconds): 31009.16
        Percent of CPU this job got: 194%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 44:59:57
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 6634452
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 3214
        Minor (reclaiming a frame) page faults: 1657511168
        Voluntary context switches: 3234186
        Involuntary context switches: 1135065
        Swaps: 0
        File system inputs: 1148048
        File system outputs: 907592
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0


## eng402001

### Configuration
Apple Macbook Pro 14"
Processor: Apple M4 Pro 14 core (10 performance 4 efficiency)
RAM:  48GB
Storage: 1TB NVME SSD
Video: Metal Support:  Metal 3
GPU: 20 cores

### Simulation Accuracy
Current Train: [0.98301667 0.01698333 0.        ]
   Best Train: [9.83316667e-01 1.66833333e-02 0.00000000e+00 6.64000000e+02]
 Current Test: [0.9652 0.0348 0.    ]
    Best Test: [9.669e-01 3.310e-02 0.000e+00 6.370e+02]

### Simulation Time
python MozafariDeepDriver.py  172501.26s user 65171.78s system 73% cpu 89:14:00.57 total

## eng402347

### Configuration
Apple Mac Mini
Processor:  Apple M1 8 cores (4 performance 4 efficiency)
RAM: 16GB
Storage: 1TB NVME SSD
Metal Support: Metal 3
GPU: 8 cores

### Simulation Accuracy
In progress...
Current Train: [0.97318333 0.02681667 0.        ]
   Best Train: [9.73183333e-01 2.68166667e-02 0.00000000e+00 1.11000000e+02]
 Current Test: [0.9595 0.0405 0.    ]
    Best Test: [9.595e-01 4.050e-02 0.000e+00 1.110e+02]
Epoch #: 112

### Simulation Time
In progress...
Today is Wednesday May 28th and the simulation has been running since May 22nd and is is only on epoch 112 of 680.

## Wind

### Configuration
Dell 15" XPS

Processor: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz with 6 cores
Memory: 32 GB
Storage: 1TB NVME SSD
GPU: Intel (with NVidia but CUDA note working correctly)

### Simulation Accuracy
Current Train: [0.97465 0.02535 0.     ]
   Best Train: [9.74933333e-01 2.50666667e-02 0.00000000e+00 2.31000000e+02]
 Current Test: [0.9612 0.0388 0.    ]
    Best Test: [9.612e-01 3.880e-02 0.000e+00 2.360e+02]
Epoch #: 237


### Simulation Time
In progress...
Today is Wednesday May 28th and the simulation has been running since May 22nd and is is only on epoch 237 of 680.

## COEN-CASSIA
### Configuration

OS: RedHat Linux
Memory: 768GB
GPU: NVIDIA Corporation AD102GL [L40S] (rev a1)
CPU: AMD EPYC 9554 64-Core Processor x 2

### Simulation Accuracy
Current Train: [0.98378333 0.01621667 0.        ]
   Best Train: [9.84633333e-01 1.53666667e-02 0.00000000e+00 6.66000000e+02]
 Current Test: [0.9658 0.0342 0.    ]
    Best Test: [9.674e-01 3.260e-02 0.000e+00 6.590e+02]


### Simulation Time
real    520m12.241s
user    2185m30.134s
sys     33m35.730s
[lukehindman@coen-cassia SpykeTorch]$ ∂