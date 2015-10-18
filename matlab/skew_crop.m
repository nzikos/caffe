clear all;
%for i=1:300
x=load('ILSVRC2014_train_00005977_1.mat');

im=x.object.data;
clear x;

im=imresize(im,[256 256]);

rot_angle_bounds = [-20 20];
theta = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);

%theta = 0.22164845699125;

if (rand(1,1)>0.5)
    randomValue1 = (-0.002 + (0.0047)*rand(1,1));
    randomValue2 = (-0.0004 + (0.00154)*rand(1,1));
else
    randomValue1 = (-0.0004 + (0.00154)*rand(1,1));
    randomValue2 = (-0.002 + (0.0047)*rand(1,1));
end
randomValue1 = 0.0027;
randomValue2 = -0.002;
theta=-20;

tform = projective2d([cosd(theta) -sind(theta) randomValue1;
                      sind(theta)  cosd(theta) randomValue2;
                      0            0                      1
                     ]);
tic;
tmp=imwarp(im,tform);
toc;
%create a mask
mask=uint8(255*im2bw(sum(tmp,3)));
%figure;imshow(mask);

[x,y]=find(mask~=255);
[a,b]=find(mask==255);
x=uint16(x);
y=uint16(y);
if(~isempty(x)) %crop if there are blanks
    tic;
    ff=round(mean(a));
    dd=round(mean(b));
    [x1, x2,x1_bound,x2_bound] = find_bounds(x,ff,size(mask,1));
    [y1, y2,y1_bound,y2_bound] = find_bounds(y,dd,size(mask,2));
    toc;    
    tmp(x1,:,1,1)=255;
    tmp(x1,:,2,1)=0;
    tmp(x1,:,3,1)=0;
    tmp(x2,:,1,1)=255;
    tmp(x2,:,2,1)=0;
    tmp(x2,:,3,1)=0;
    tmp(:,y1,1,1)=255;   
    tmp(:,y1,2,1)=0;
    tmp(:,y1,3,1)=0;
    tmp(:,y2,1,1)=255;
    tmp(:,y2,2,1)=0;
    tmp(:,y2,3,1)=0;
    
    tmp(x1_bound,:,1,1)=0;
    tmp(x1_bound,:,2,1)=0;
    tmp(x1_bound,:,3,1)=255;
    tmp(x2_bound,:,1,1)=0;
    tmp(x2_bound,:,2,1)=0;
    tmp(x2_bound,:,3,1)=255;
    tmp(:,y1_bound,1,1)=0;   
    tmp(:,y1_bound,2,1)=0;
    tmp(:,y1_bound,3,1)=255;
    tmp(:,y2_bound,1,1)=0;
    tmp(:,y2_bound,2,1)=0;
    tmp(:,y2_bound,3,1)=255;
    
    mask=repmat(mask,1,1,3);
    mask=mask+100;
    mask(x1,:,1)=255;mask(x1,:,2)=0;mask(x1,:,3)=0;
    mask(x2,:,1)=255;mask(x2,:,2)=0;mask(x2,:,3)=0;
    mask(:,y1,1)=255;mask(:,y1,2)=0;mask(:,y1,3)=0;
    mask(:,y2,1)=255;mask(:,y2,2)=0;mask(:,y2,3)=0;
    
    mask(x1_bound,:,1)=0;mask(x1_bound,:,2)=0;mask(x1_bound,:,3)=255;
    mask(x2_bound,:,1)=0;mask(x2_bound,:,2)=0;mask(x2_bound,:,3)=255;
    mask(:,y1_bound,1)=0;mask(:,y1_bound,2)=0;mask(:,y1_bound,3)=255;
    mask(:,y2_bound,1)=0;mask(:,y2_bound,2)=0;mask(:,y2_bound,3)=255;    
    
    mask(round(mean(a)),:,1)=255;
    mask(round(mean(a)),:,2)=0;
    mask(round(mean(a)),:,3)=255;
    mask(:,round(mean(b)),1)=255;
    mask(:,round(mean(b)),2)=0;
    mask(:,round(mean(b)),3)=255;    
    mask(:,round(size(mask,2)/2),1)=255;
    mask(:,round(size(mask,2)/2),2)=255;
    mask(:,round(size(mask,2)/2),3)=0;
    mask(round(size(mask,1)/2),:,1)=255;
    mask(round(size(mask,1)/2),:,2)=255;
    mask(round(size(mask,1)/2),:,3)=0;    
end
imshow(mask);
% pause(8);
% end