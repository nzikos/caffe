clear all;
im=imread('n01514668_359.JPEG');
theta=-15;

batch(:,:,:,1)=im;
batch(:,:,:,2)=im;

% tform = affine2d([cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1]);
% 
% batch2=imwarp(batch,tform);
% figure,imshow(batch2(:,:,:,1));
% figure,imshow(batch2(:,:,:,2));
% crop_h = abs(ceil(sind(theta)*size(batch,2)));
% crop_w = abs(ceil(sind(theta)*size(batch,1)));
% 
% batch3=batch2(1+crop_h:end-crop_h,1+crop_w:end-crop_w,:,:);
batch2=imrotate(batch,theta);
batch3=imrotate(batch,theta,'crop');
figure,imshow(batch(:,:,:,1));
figure,imshow(batch2(:,:,:,1));
figure,imshow(batch3(:,:,:,2));