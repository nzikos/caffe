% Close figures.
delete(findall(0,'Type','figure'))

load('ILSVRC2012_val_00001047_1.mat')

im=permute(object.data,[2 1 3]);
dims=[227 227];
object_dims=[256 256];
im=imresize(im,object_dims);

rnd_height_1 = uint32((object_dims(1)-dims(1))*rand());
rnd_width_1  = uint32((object_dims(2)-dims(2))*rand());
if rnd_height_1==0
    rnd_height_1=rnd_height_1+1;
end
if rnd_width_1==0
    rnd_width_1=rnd_width_1+1;
end
rnd_height_2 = rnd_height_1+dims(1);
rnd_width_2 = rnd_width_1+dims(2);

figure;imshow(im);
figure;imshow(flip(im,2));

new_im = im;

new_im(1:rnd_height_1,:,:)      = new_im(1:rnd_height_1,:,:) - 30;
new_im(rnd_height_2:end,:,:)    = new_im(rnd_height_2:end,:,:) -30;
new_im(1:end,1:rnd_width_1,:)   = new_im(1:end,1:rnd_width_1,:)-30;
new_im(1:end,rnd_width_2:end,:) = new_im(1:end,rnd_width_2:end,:)-30;

new_im(rnd_height_1,:,1)=255;
new_im(rnd_height_1,:,2)=0;
new_im(rnd_height_1,:,3)=0;
new_im(:,rnd_width_1,1)=255;
new_im(:,rnd_width_1,2)=0;
new_im(:,rnd_width_1,3)=0;
new_im(rnd_height_2,:,1)=255;
new_im(rnd_height_2,:,2)=0;
new_im(rnd_height_2,:,3)=0;
new_im(:,rnd_width_2,1)=255;
new_im(:,rnd_width_2,2)=0;
new_im(:,rnd_width_2,3)=0;


figure;imshow(new_im);

new_im = im(rnd_height_1+1:rnd_height_2,rnd_width_1+1:rnd_width_2,:);

figure;imshow(new_im)

theta = 30;

R=[cosd(theta) -sind(theta) 0; +sind(theta) cosd(theta) 0; 0 0 1];

shear_v = sind(0); shear_h = 0.35;
S=[1       shear_v 0 ;
   shear_h 1       0 ;
   0       0      1];

tform = affine2d(S);

new_im = imwarp(im,tform);
new_im2 = new_im;

u = [1 1 ;1 size(im,1);size(im,2) 1; size(im,2) size(im,1)];
v = tform.transformPointsForward(u);
v(:,1)=(v(:,1)-min(v(:,1)))+1;
v(:,2)=(v(:,2)-min(v(:,2)))+1;
%v=ceil(v)+1
v=ceil(v);

new_im3 = crop_blanks(new_im,tform);
%print transformed coordinates on image
new_im(v(1,2),v(1,1),1)=255;
new_im(v(1,2),v(1,1),2)=0;
new_im(v(1,2),v(1,1),3)=0;
new_im(v(2,2),v(2,1),1)=255;
new_im(v(2,2),v(2,1),2)=0;
new_im(v(2,2),v(2,1),3)=0;
new_im(v(3,2),v(3,1),1)=255;
new_im(v(3,2),v(3,1),2)=0;
new_im(v(3,2),v(3,1),3)=0;
new_im(v(4,2),v(4,1),1)=255;
new_im(v(4,2),v(4,1),2)=0;
new_im(v(4,2),v(4,1),3)=0;

min_v_h = sort(v(:,1));
height_1_bound = floor(abs(min_v_h(1)-min_v_h(2))/2)+1;
height_2_bound = floor(abs(min_v_h(3)-min_v_h(4))/2) + min_v_h(3);
min_v_w = sort(v(:,2));
width_1_bound = floor(abs(min_v_w(1)-min_v_w(2))/2)+1;
width_2_bound = floor(abs(min_v_w(3)-min_v_w(4))/2) + min_v_w(3);

new_im(width_1_bound,:,1)=255;
new_im(width_1_bound,:,2)=0;
new_im(width_1_bound,:,3)=0;
new_im(width_2_bound,:,1)=255;
new_im(width_2_bound,:,2)=0;
new_im(width_2_bound,:,3)=0;

new_im(:,height_1_bound,1)=255;
new_im(:,height_1_bound,2)=0;
new_im(:,height_1_bound,3)=0;
new_im(:,height_2_bound,1)=255;
new_im(:,height_2_bound,2)=0;
new_im(:,height_2_bound,3)=0;

for i=1:size(new_im,2)
    for j=1:width_1_bound-1
        for k=1:3
            if new_im(j,i,k)==0
                new_im(j,i,k)=new_im(j,i,k)+100;
            else
                if new_im(j,i,k)~=100
                    new_im(j,i,k)=new_im(j,i,k)-100;
                end
            end
        end
    end
end
for i=1:height_1_bound-1
    for j=1:size(new_im,1)
        for k=1:3
            if new_im(j,i,k)==0
                new_im(j,i,k)=new_im(j,i,k)+100;
                
            else
                if new_im(j,i,k)~=100
                    new_im(j,i,k)=new_im(j,i,k)-100;
                end
            end            
        end
    end
end
for i=height_2_bound+1:size(new_im,2)
    for j=1:size(new_im,1)
        for k=1:3
            if new_im(j,i,k)==0
                new_im(j,i,k)=new_im(j,i,k)+100;
            else
                if new_im(j,i,k)~=100
                    new_im(j,i,k)=new_im(j,i,k)-100;
                end
            end
        end
    end
end
for i=1:size(new_im,2)
    for j=width_2_bound+1:size(new_im,1)
        for k=1:3
            if new_im(j,i,k)==0
                new_im(j,i,k)=new_im(j,i,k)+100;
            else
                if new_im(j,i,k)~=100
                    new_im(j,i,k)=new_im(j,i,k)-100;
                end
            end
        end        
    end
end
figure;imshow(new_im);
new_im = new_im2(width_1_bound:width_2_bound,height_1_bound:height_2_bound,:);
new_im = imresize(new_im,[256 256]);
figure;imshow(new_im);
new_im = new_im/2 + im/2;
figure;imshow(new_im);