function out = async_read_from_disk_and_preprocess(x)
%ASYNC_READ_FROM_DISK_AND_PREPROCESS  This function is used by batch_factory
% in order to load asynchronously the batch from the hard disk and process
% the objects before network feeds them to caffe.

%% Processing details
% Since CNNs are invariant in many transformations, I.e. if a neural net
% sees for the first time a human face rotated by 30 degrees it may not
% recognize it, certain affine transformations are applied creating much
% more unique images for the neural net, to learn. Such transformations are:
%
% 1. The translation transformation which is cutting a random box,
%    whose dimensions are the neural net's input dimensions, from the
%    object, whose dimensions MUST be greater than the above dimensions.
%
% 2. Horizontal mirroring
%
% 3. The rotation transformation which rotates the objects by random
%    degrees per batch.
%
% MORE TO BE ADDED...

%% Argument x is a struct which contains:
%
% paths              : The paths of the objects that'll be loaded as a Cell Array of
%                      strings.
%
% input_dims         : The dimensions of caffe input layer.
%
% use_flipped        : use horizontally flipped objects or not (0/1).
%
% crop_freq          : Ratio of objects random cropping per batch.
%
% rot_ratio          : The ratio of object rotation inside the batch
%
% rot_theta_bounds   : The bounds of the random theta during rotation in degrees.
%
%% Values returned
% data    : The objects/images as HxWx3xN (where N is the batch size)
% uids    : The unique ids of the class as defined from the extraction_model
%           matrix with dimensions of 1x1xKxN
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

%% DO NOT DELETE THIS LINE. BAD THINGS WILL HAPPEN
% Bug found under ubuntu 14.04 / Matlab R2015a
% Workers garbage collector is not cleaning the memory efficiently which
% causes extensive memory usage for workers making the system unresponsive 
% while calling parfeval several times from within a loop.
java.lang.System.gc(); %FATAL MEMORY LEAK SOLVED

%% PREPARE ATTRIBUTES
paths        = x.paths;
input_dims   = [x.input_dims.width x.input_dims.height]; %Remember inconsistency between matlab and caffe dimensions

use_flipped  = x.use_flipped;

crop_freq    = x.crop_freq;
rots_freq    = x.rot_freq;
skew_freq    = x.skew_freq;
projs_freq   = x.projections_freq;

rot_theta_bounds  = x.rot_theta_bounds;
skew_bounds = x.skew_bounds;

batch_size = length(paths);
%% EXTRACT OBJECTS
for i=batch_size:-1:1
    tmp = load(paths{i});
    data(:,:,:,i)=tmp.object.data;
    uids(i,:)=tmp.object.uid;
end

%% Convert frequencies to batch regions
inner_counter = 0;

crops_reg         = [inner_counter+1 ; inner_counter+round(batch_size*crop_freq)];
inner_counter     = inner_counter +round(batch_size*crop_freq);

rots_reg          = [inner_counter+1 ; inner_counter+round(batch_size*rots_freq)];
inner_counter     = inner_counter +round(batch_size*rots_freq);

skew_reg          = [inner_counter+1 ; inner_counter+round(batch_size*skew_freq)];
inner_counter     = inner_counter +round(batch_size*skew_freq);

projs_reg         = [inner_counter+1 ; inner_counter+round(batch_size*projs_freq)];
inner_counter     = inner_counter +round(batch_size*projs_freq);

non_processed_reg = [inner_counter+1 ; batch_size];

%% PERFORM PROCESSING
if crops_reg(1)<=crops_reg(2)
    object_dims = size(data);    
    if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
        processed_data(:,:,:,crops_reg(1):crops_reg(2)) = get_random_crops(data(:,:,:,crops_reg(1):crops_reg(2)),input_dims);
    else
        APP_LOG('error',0,'Identical dimensions detected while extracting random segments');
        APP_LOG('error',0,'Object dimensions are [%d,%d]',object_dims(1),object_dims(2));
        APP_LOG('last_error',0,'Network dimensions are [%d,%d]',input_dims(1),input_dims(2));
    end
end

if rots_reg(1)<=rots_reg(2)
    processed_data(:,:,:,rots_reg(1):rots_reg(2)) = get_random_rotations(rot_theta_bounds,data(:,:,:,rots_reg(1):rots_reg(2)),input_dims);
end

if skew_reg(1)<=skew_reg(2)
    processed_data(:,:,:,skew_reg(1):skew_reg(2)) = get_random_skews(skew_bounds,data(:,:,:,skew_reg(1):skew_reg(2)),input_dims);
end

if projs_reg(1)<=projs_reg(2)
    processed_data(:,:,:,projs_reg(1):projs_reg(2))=get_projections(data(:,:,:,projs_reg(1):projs_reg(2)),input_dims,rot_theta_bounds);
end

if non_processed_reg(1)<=non_processed_reg(2)
    processed_data(:,:,:,non_processed_reg(1):non_processed_reg(2)) = get_resized_objects(data(:,:,:,non_processed_reg(1):non_processed_reg(2)),input_dims);
end
clear data;

%% PERFORM HORIZONTAL FLIPPING
if(use_flipped)
    rnd_flip_idxs=randperm(batch_size,floor(0.5*batch_size));
    processed_data(:,:,:,rnd_flip_idxs)=flip(processed_data(:,:,:,rnd_flip_idxs),1); %Width is 1st dimension, so we are actually performing vertical flip on a 90degrees rotated object
end

%% OUTPUT
out{1,1}=processed_data;
out{2,1}=int32(uids);
end

function output_data = get_projections(data,dims,rot_theta_bounds)

theta = rot_theta_bounds(1) + (rot_theta_bounds(2)-rot_theta_bounds(1))*rand(1,1);

if (rand(1,1)>0.5)
    randomValue1 = (-0.002 + (0.0047)*rand(1,1));
    randomValue2 = (-0.0004 + (0.00154)*rand(1,1));
else
    randomValue1 = (-0.0004 + (0.00154)*rand(1,1));
    randomValue2 = (-0.002 + (0.0047)*rand(1,1));
end

tform = projective2d([cosd(theta) -sind(theta) randomValue1;
                      sind(theta)  cosd(theta) randomValue2;
                      0            0                      1
                     ]);
                 
tmp=imwarp(data,tform);

%create a mask
mask=uint8(255*im2bw(sum(sum(tmp,4),3)));
[x,y]=find(mask~=255);
[a,b]=find(mask==255);
x=uint16(x);
y=uint16(y);
a=round(mean(a)); %Centroid - X - of mask
b=round(mean(b)); %Centroid - Y - of mask

if(~isempty(x)) %crop if there are blanks
    [x1, x2] = find_bounds(x,a,size(mask,1));
    [y1, y2] = find_bounds(y,b,size(mask,2));
    output_data = get_resized_objects(tmp(x1:x2,y1:y2,:,:),dims);    
else
    output_data = get_resized_objects(tmp,dims);        
end
end

function [bound_1, bound_2] = find_bounds(x,k1,siz)
     
    x_distr_tmp=int16(tabulate(x));

    x_distr=int32(zeros(siz,2));
    x_distr(:,1)=1:siz;
    x_distr(x_distr_tmp(:,1),2)=x_distr_tmp(:,2);
    
    x_der=int32(zeros(siz,1));
    x_2nd_der=int32(zeros(siz,1));
    for i=2:8
        x_der(i:end) = x_der(i:end) + x_distr(i:end,2)-x_distr(1:end-i+1,2);
    end
    for i=2:8
        x_2nd_der(i:end) = x_2nd_der(i:end) + x_der(i:end)-x_der(1:end-i+1);
    end

    
    %find first maximum
    [~,x1_bound]           = max(x_2nd_der(1:k1));   
    x1_bound=x1_bound-4;
    if x1_bound<1
        x1_bound=1;
    end
    
    [~,x2_bound]           = max(x_2nd_der(k1+1:end));
    x2_bound=x2_bound+k1-1-4;
            
    bound_1 =round(x1_bound/2);
    bound_2 =round(x2_bound + (siz - x2_bound)/2);
end
