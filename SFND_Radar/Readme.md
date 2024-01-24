### Udacity Radar 
Matlab version used for implementation
```
Matlab2023b
```

#### FMCW Waveform Design
With the following chirp design parameters a slope of 2.05e13 is achived:
```
B = c / (2*spec.range_res);
Td_max = 2*(spec.range_max/c);
Tchirp = 5.5*Td_max;
slope = B/Tchirp;
```
#### 1D FFT FOR RANGE ESTIMATION
![Range](https://github.com/danny-bit/udacity_sensorfusion/assets/59084863/66dedd0b-6c64-4964-b393-19534d6f7f1b)

#### CFAR2D - Implementation steps
- The 2D CFAR approach implemented is using cell averaging.
For the implementation a convolution approach was used, as this can be vectorized.
The kernel was calculated using the given training and guard cells.
After convolution with this kernel a dynamic thershold is obtained.
For thresholding an additional offset is added.
```
thres_dB = pow2db(conv2(db2pow(RDM_c),kernel, 'same'));
thres_dB = thres_dB + offset_dB;
```

![2024-01-24 17_58_35-Figure 7](https://github.com/danny-bit/udacity_sensorfusion/assets/59084863/dc104e36-a583-4e79-977b-d7b8771b8995)

- Selection of Training, Guard cells and offset.
Empirically the following parameters were obtained:

```
nTrainCells = 12;
nTrainBands = 8;
nGuardCells = 4;
nGuardBands = 4;
```
- Steps taken to suppress the non-thresholded cells at the edges.
For the thresholding on the edges padding was used.
The original FFT was paded, whereby the edge values where hold.

```
% pad edges
oX = nTrainBands+nGuardBands;
oY = nTrainCells+nGuardCells;
RDM_c = paddata(RDM,size(RDM,1)+2*oX, 'Pattern','edge','Side','both');
RDM_c = paddata(RDM_c',size(RDM,2)+2*oY, 'Pattern','edge','Side','both')';
```
