% test the Image Deconvolution using Variational Method
% function fTestImageLRTV
%%
clear;clc;
addpath mylib
n = 256;
% name = 'lena';
% name = 'xch2';
name = 'sperm'

f0 = load_image(name);
% if f0 is a color image (dim > 2), only choose the first channel as GRAY
% Image
if ndims(f0) > 2 ; f0 = f0(:,:,1); end;

% resize or filling the image to the size of n x n;
% if any dim of the image >= n, resize or cut the dim
% or creating a n x n size 0 matrix; and filling the f0 into the 0 matrix
if min(size(f0)) >= n
    f0 = rescale(crop(f0,n));
else
    img = zeros(n,n);
    img(1:size(f0,1),1:size(f0,2)) = f0;
    f0=img;
    clear img;
end
org=f0;
rate = 2;

%% preprocessing
s = 1;
% n is the number of image row
n=size(f0,1);
% generate a vector from 0 to n/2 - 1 and from -n/2 to -1
x = [0:n/2-1, -n/2:-1];
% generate a mesh gride coordinate matrix X and Y
[Y,X] = meshgrid(x,x);
% using Gaussian func to calculate a 2D filter h
h = exp( (-X.^2-Y.^2)/(2*s^2) );
% normorlize filter h
h = h/sum(h(:));

% define a func Phi which using filtering methods (multi category) to
% filter x
Phi = @(x,h)real(ifft2(fft2(x).*fft2(h)));
% using filter h to filter f0
y0 = Phi(f0,h);

% downsampling for the filtered y0; rate = 2
ylr = my_downsample(y0,rate);
% show downsampling image ylr
figure(1);imshow(ylr,[]);
% upsampling for the downsampled image ylr
g = my_upsample(ylr,rate);
% resize upsampled image g, make sure the size of g is the same as y0
g=g(1:size(y0,1),1:size(y0,2)); 
% show upsampled image g
figure(2);imshow(g,[]);
% calculte the SNR from original image f0 and the upsampled image g
snr(f0,g)
% show original image f0
figure(3);imshow(f0,[]);

% input is original and output is larger
%% image recover
para.niter = 6;
para.dt = 0.1;

[fTV, errList_H] = fTestImageLRTV2D(ylr,rate,f0,para);
snr(f0,fTV)
figure(4);imshow(fTV,[]);

