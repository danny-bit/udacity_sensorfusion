#### Udacity Radar 



Matlab version used for implementation
```
Matlab2023b
```
- Implementation steps for the 2D CFAR process.
The 2D CFAR approach implemented is useing cell averaging.
For the implementation a convolution approach was used, as this can be vectorized.
The kernel was calculated using the given training and guard cells.
After convolution with this kernel a dynamic thershold is obtained.
For thresholding an additional offset is added.

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
