close all;
im0 = imread('Proj3.tif');
figure 
imshow(im0);
title("Original Image");

%take the fouier transform of the image
im1 = fftshift(fft2(im0));

%figure
%imagesc(log(1+abs(im1)));
%colormap gray

%from fft, identify high intensity points:
% 1. [280, 219]
% 2. [282, 196]
% 3. [275, 182]

%create bandpass filter
[n,m,~] = size(im0);
imfilt = zeros(n,m);
imfilt(191,266) = 1;
imfilt(214,264) = 1;
imfilt(182,275) = 1;
imfilt(196,282) = 1;
imfilt(219,280) = 1;
imfilt(228,271) = 1;
imfilt = logical(imfilt);

figure 
imshow(imfilt);
title("Frequency Domain Filter");

%show bandpass filter
fftfilter = ifft2(fftshift(imfilt));
figure 
imagesc(log(1+abs(fftfilter)));
colormap gray
title("Extracted Periodic Pattern");

%apply filter to image
im2 = im1;
im2(imfilt) = 0;

%figure
%imagesc(log(1+abs(im2)));
%colormap gray

%inverse transform
im3 = ifft2(fftshift(im2));

figure
imagesc(1+abs(im3));
colormap gray
title("Filtered Image");



%Removing impact of non-uniform illumination
se = strel('disk', 10);
background = imopen(im0,se);
%figure
%imshow(background);

im4 = im0 - background;
%figure
%imshow(im4);

im5 = imadjust(im4,[0 0.2]);
figure
imshow(im5);
title("Impact of non-uniform illumination removed using strel");
colormap gray