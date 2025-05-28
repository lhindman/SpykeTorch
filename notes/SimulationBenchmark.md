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

