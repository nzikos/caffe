%clear all;
for i=1:100000
%x=load('ILSVRC2012_val_00001154_6.mat');

im=256*ones(256,256);
%clear x;

im=imresize(im,[256 256]);

rot_angle_bounds = [-20 20];
shear_percentage = [0 0.4];

theta   = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);
shear_h = shear_percentage(1) + (shear_percentage(2)-shear_percentage(1))*rand(1,1);
shear_v = shear_percentage(1) + (shear_percentage(2)-shear_percentage(1))*rand(1,1);

if (rand(1,1)>0.5)
    randomValue1 = (-0.002 + (0.0047)*rand(1,1));
    randomValue2 = (-0.0004 + (0.00154)*rand(1,1));
else
    randomValue1 = (-0.0004 + (0.00154)*rand(1,1));
    randomValue2 = (-0.002 + (0.0047)*rand(1,1));
end

% theta =5;
% shear_h =0.4;
% shear_v =0.2;

tform = projective2d([cosd(theta) -sind(theta) randomValue1;
                       sind(theta)  cosd(theta) randomValue2;
                       0       0       1]);
tic;
tmp=imwarp(im,tform);
toc;
u = [1 1 ;1 size(im,1);size(im,2) 1; size(im,2) size(im,1)];
v = tform.transformPointsForward(u);
v(:,1)=(v(:,1)-min(v(:,1)))+1;
v(:,2)=(v(:,2)-min(v(:,2)))+1;
v=ceil(v);

%print transformed coordinates on image
tmp(v(1,2),v(1,1),1)=255;
tmp(v(1,2),v(1,1),2)=0;
tmp(v(1,2),v(1,1),3)=0;
tmp(v(2,2),v(2,1),1)=255;
tmp(v(2,2),v(2,1),2)=0;
tmp(v(2,2),v(2,1),3)=0;
tmp(v(3,2),v(3,1),1)=255;
tmp(v(3,2),v(3,1),2)=0;
tmp(v(3,2),v(3,1),3)=0;
tmp(v(4,2),v(4,1),1)=255;
tmp(v(4,2),v(4,1),2)=0;
tmp(v(4,2),v(4,1),3)=0;

min_v_h = sort(v(:,1));
height_1_bound = floor(abs(min_v_h(1)-min_v_h(2))/2)+1;

height_2_bound = floor(abs(min_v_h(3)-min_v_h(4))/2)+1 + min_v_h(3);

tmp(:,height_1_bound,:)=255;
tmp(:,height_2_bound,:)=255;

min_v_w = sort(v(:,2));
width_1_bound = floor(abs(min_v_w(1)-min_v_w(2))/2)+1;
width_2_bound = floor(abs(min_v_w(3)-min_v_w(4))/2)+1 + min_v_w(3);

tmp(width_1_bound,:,:)=255;
tmp(width_2_bound,:,:)=255;

imshow(tmp(width_1_bound:width_2_bound,height_1_bound:height_2_bound,:));
end
%% DEPRECATED
%create a mask
% %mask=uint8(255*im2bw(sum(tmp,3)));
% mask_tmp=uint8(255*ones(size(im,1),size(im,2),1));
% imshow(mask_tmp);figure;
% mask = imwarp(mask_tmp,tform);
% %figure;imshow(mask);
% 
% [x,y]=find(mask~=255);
% [a,b]=find(mask==255);
% x=uint16(x);
% y=uint16(y);
% if(~isempty(x)) %crop if there are blanks
%     tic;
%     ff=round(mean(a));
%     dd=round(mean(b));
%    [x1, x2,x1_bound,x2_bound] = find_bounds(x,ff,size(mask,1));
%    [y1, y2,y1_bound,y2_bound] = find_bounds(y,dd,size(mask,2));
%    toc;    
%     tmp(x1,:,1,1)=255;
%     tmp(x1,:,2,1)=0;
%     tmp(x1,:,3,1)=0;
%     tmp(x2,:,1,1)=255;
%     tmp(x2,:,2,1)=0;
%     tmp(x2,:,3,1)=0;
%     tmp(:,y1,1,1)=255;   
%     tmp(:,y1,2,1)=0;
%     tmp(:,y1,3,1)=0;
%     tmp(:,y2,1,1)=255;
%     tmp(:,y2,2,1)=0;
%     tmp(:,y2,3,1)=0;
%     
%     tmp(x1_bound,:,1,1)=0;
%     tmp(x1_bound,:,2,1)=0;
%     tmp(x1_bound,:,3,1)=255;
%     tmp(x2_bound,:,1,1)=0;
%     tmp(x2_bound,:,2,1)=0;
%     tmp(x2_bound,:,3,1)=255;
%     tmp(:,y1_bound,1,1)=0;   
%     tmp(:,y1_bound,2,1)=0;
%     tmp(:,y1_bound,3,1)=255;
%     tmp(:,y2_bound,1,1)=0;
%     tmp(:,y2_bound,2,1)=0;
%     tmp(:,y2_bound,3,1)=255;
%     
%     mask=repmat(mask,1,1,3);
%     mask=mask+100;
%     mask(x1,:,1)=255;mask(x1,:,2)=0;mask(x1,:,3)=0;
%     mask(x2,:,1)=255;mask(x2,:,2)=0;mask(x2,:,3)=0;
%     mask(:,y1,1)=255;mask(:,y1,2)=0;mask(:,y1,3)=0;
%     mask(:,y2,1)=255;mask(:,y2,2)=0;mask(:,y2,3)=0;
%     
%     mask(x1_bound,:,1)=0;mask(x1_bound,:,2)=0;mask(x1_bound,:,3)=255;
%     mask(x2_bound,:,1)=0;mask(x2_bound,:,2)=0;mask(x2_bound,:,3)=255;
%     mask(:,y1_bound,1)=0;mask(:,y1_bound,2)=0;mask(:,y1_bound,3)=255;
%     mask(:,y2_bound,1)=0;mask(:,y2_bound,2)=0;mask(:,y2_bound,3)=255;    
%     
%     mask(round(mean(a)),:,1)=255;
%     mask(round(mean(a)),:,2)=0;
%     mask(round(mean(a)),:,3)=255;
%     mask(:,round(mean(b)),1)=255;
%     mask(:,round(mean(b)),2)=0;
%     mask(:,round(mean(b)),3)=255;    
%     mask(:,round(size(mask,2)/2),1)=255;
%     mask(:,round(size(mask,2)/2),2)=255;
%     mask(:,round(size(mask,2)/2),3)=0;
%     mask(round(size(mask,1)/2),:,1)=255;
%     mask(round(size(mask,1)/2),:,2)=255;
%     mask(round(size(mask,1)/2),:,3)=0;    
% end
% imshow(mask);
% pause(8);
% end