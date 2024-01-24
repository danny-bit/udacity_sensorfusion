clear all;  clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%
spec.fc = 77e9;
spec.range_max = 200;
spec.range_res = 1;
spec.v_max = 100;

% Nd ... number of chirps in one sequence
% Nr ... number of samples on each chirp. 
Nd=128;  % #of doppler cells OR #of sent periods % number of chirps
Nr=1024; %for length of time OR # of range cells

% constants
c = 3e8; % approx. speed of light

%% User Defined Range and Velocity of target

x0_target = 110; % [m]   permissible range: [10 to 200]
v0_target = 30;  % [m/s] permissible range: [-70 to 70]

%% FMCW Waveform Generation
% FMCW: Calculate chirp characteristics from requirements
B = c / (2*spec.range_res);

% chirp time Ts should be 5-6 times the roundtrip time, such that
% td/Ts < 0.2;
Td_max = 2*(spec.range_max/c);
Tchirp = 5.5*Td_max;
slope = B/Tchirp;

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples

x_target = x0_target + v0_target*t;
Td = 2*x_target/c;

fprintf('slope: %0.2e\n', B/Tchirp)
%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

r_t = x0_target+v0_target*t;
td  = 2*r_t/c;
phase_tx = spec.fc*t+slope*t.^2/2;
phase_rx = spec.fc*(t-td)+slope*(t-td).^2/2;
Tx = cos(2*pi*phase_tx);
Rx = cos(2*pi*phase_rx);

% frequency mixing by multiplication
Mix = Rx.*Tx; 

%% RANGE MEASUREMENT

% reshape the vector into Nr*Nd array. 
% Nr and Nd here would also define the size of Range and Doppler FFT 
% respectively.
Mix = reshape(Mix, [Nr,Nd]);

%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
Mix_fft = fft(Mix, Nr, 1);

% Take the absolute value of FFT output
Mix_fft_abs = abs(Mix_fft);

% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
Mix_fft_abs  = Mix_fft_abs(1:(Nr/2-1),:);
Mix_fft_norm = Mix_fft_abs./(ones(Nr/2-1,1)*max(Mix_fft_abs));

%return
%plotting the range
figure ('Name','Range from First FFT', ...
        'Color','white')
plot(Mix_fft_norm);
ylabel('Magnitude Norm');
xlabel('Range [m]');
axis ([0 200 0 1]);
title('Range from First FFT')
grid on;

%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure('Color','white');
surf(doppler_axis,range_axis,RDM);
title('Range Doppler Map')
%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% CFAR configuration
% cells (doppler direction) / bands (range direction)
nTrainCells = 12;
nTrainBands = 8;
nGuardCells = 4;
nGuardBands = 4;

% offset the threshold by SNR value in dB
offset_dB = 5.5;

%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);

Nx = 2*(nTrainCells+nGuardCells)+1; Ny = 2*(nTrainBands+nGuardBands)+1;
cX = ceil(Nx/2);
cY = ceil(Ny/2);
oX = nTrainBands+nGuardBands;
oY = nTrainCells+nGuardCells;

% create CFAR kernel
kernel = ones(Ny,Nx);
kernel(cY, cX) = 0;
kernel(cY-nGuardBands:cY+nGuardBands,cX-nGuardCells:cX+nGuardCells)=0;
kernel=1/(sum(sum(kernel)))*kernel;

% pad edges
RDM_c = paddata(RDM,size(RDM,1)+2*oX, ...
                'Pattern','edge','Side','both');
RDM_c = paddata(RDM_c',size(RDM,2)+2*oY,...
                'Pattern','edge','Side','both')';

% calc threshold by convolution % remove padding
thres_dB = pow2db(conv2(db2pow(RDM_c),kernel, 'same'))+offset_dB;
thres_dB = thres_dB(oX:oX+size(RDM,1)-1,oY:oY+size(RDM,2)-1);

% apply threshold
out = RDM>thres_dB;

% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure('Color', 'white');
surf(doppler_axis,range_axis,double(out));
xlabel('Range [m]')
ylabel('Radial velocity [m/s]')
colorbar;


 
 