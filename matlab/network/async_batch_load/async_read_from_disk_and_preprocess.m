function out = async_read_from_disk_and_preprocess(x)
%ASYNC_READ_FROM_DISK_AND_PREPROCESS  This function is used by batch_factory
% in order to load asynchronously the batch from the hard disk and process
% the objects before network feeds them to caffe.

%% Processing details
% Since CNNs are invariant in many transformations, I.e. if a neural net
% sees for the first time a human face rotated by 30 degrees it may not
% recognise it, certain affine transformations are applied creating much
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
% use_flip           : Boolean value in order to flip randomly the 50% of
%                      objects.
%
% use_random_segments: Get a random box from the original object. i.e.
%                      supplying objects of 259x259 dims and getting a
%                      227x227 box. This technique forces the neural
%                      network to be translation variant.
% use_rot            : Use rotation of objects
%
% rnd_segments_ratio : The ratio of object segmentation inside the batch
%
% rot_ratio          : The ratio of object rotation inside the batch
%
% rot_angle_bounds   : The bounds of the random angle during rotation in degrees.
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
% while calling parfeval severaltimes from within a loop.
java.lang.System.gc(); %FATAL MEMORY LEAK SOLVED

%% PREPARE ATTRIBUTES
paths                = x.paths;
input_dims           = x.input_dims;

use_flip             = x.use_flip;
use_random_segments  = x.use_random_segments;
use_rot              = x.use_rot;
use_projs            = x.use_projs;


rnd_segments_ratio   = x.freq_random_segments;
rot_ratio            = x.rot_freq;
projections_ratio    = x.projections_freq;

rot_angle_bounds     = x.rot_angle_bounds;

batch_size = length(paths);
%% EXTRACT OBJECTS
for i=batch_size:-1:1
    tmp = load(paths{i});
    data(:,:,:,i)=tmp.object.data;
    uids(1,1,1,i)=tmp.object.uid;
end

%% Convert frequencies to batch regions
from = 1;
to   = 0;
if use_random_segments
    to   = to + floor(batch_size * rnd_segments_ratio);
    random_segments_region = [from to];
end
if use_rot
    from = to + 1;
    to = to + floor(batch_size * rot_ratio) -1 ;
    rots_region = [from to];
end
if use_projs
    from = to + 1;
    to = to + floor(batch_size * projections_ratio);
    projs_region = [from to];
end
from = to + 1;
to = batch_size;
non_processed_region = [from to];

%% PERFORM PROCESSING
object_dims = size(data);
if(use_random_segments)
    if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
        processed_data = get_random_segments(data(:,:,:,random_segments_region(1):random_segments_region(2)),input_dims);
    else
        APP_LOG('error',0,'Identical dimensions detected while extracting random segments');
        APP_LOG('error',0,'Object dimensions are [%d,%d]',object_dims(1),object_dims(2));
        APP_LOG('error_last',0,'Network dimensions are [%d,%d]',input_dims(1),input_dims(2));
    end
end
if (use_rot)
    theta = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);
    temp_data = imrotate(data(:,:,:,rots_region(1):rots_region(2)),theta,'bilinear','crop');
    %Resize to fit network input
     if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
         processed_data(:,:,:,rots_region(1):rots_region(2))=get_resized_objects(temp_data,input_dims);
     else
         processed_data(:,:,:,rots_region(1):rots_region(2))=temp_data;
     end
     clear temp_data;
end
if (use_projs)
    processed_data(:,:,:,projs_region(1):projs_region(2))=get_projections(data(:,:,:,projs_region(1):projs_region(2)),input_dims,rot_angle_bounds);
end
%% PASS ANY LEFTOVERS
if non_processed_region(2)>=non_processed_region(1)
    if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
        processed_data(:,:,:,non_processed_region(1):non_processed_region(2)) = get_resized_objects(data(:,:,:,non_processed_region(1):non_processed_region(2)),input_dims);
    else
        processed_data(:,:,:,non_processed_region(1):non_processed_region(2)) = data(:,:,:,non_processed_region(1):non_processed_region(2));
    end
end
clear data;

%% PERFORM HORIZONTAL FLIPPING IN ALL DATA
if(use_flip)
    rnd_flip_idxs=randperm(batch_size,floor(0.5*batch_size));
    processed_data(:,:,:,rnd_flip_idxs)=flip(processed_data(:,:,:,rnd_flip_idxs),2);
end
%% OUTPUT
out{1,1}=processed_data;
out{2,1}=uids;
end

% %% SET INDICES
% from=1;
% to=1;
% 
% object_dims = size(data);
% %% PERFORM RANDOM SEGMENTATION FOR TRANSLATION INVARIANCE
% if(use_random_segments)
%     from=1;
%     to=from+round(rnd_segments_ratio*batch_size)-1;
%     if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
%         new_data = get_random_segments(data(:,:,:,from:to),input_dims);
%     else
%         APP_LOG('error',0,'Identical dimensions detected while extracting random segments');
%         APP_LOG('error',0,'Object dimensions are [%d,%d]',object_dims(1),object_dims(2));
%         APP_LOG('error_last',0,'Network dimensions are [%d,%d]',input_dims(1),input_dims(2));
%     end
% end
% %% PERFORM ROTATION FOR ROTATION INVARIANCE
% if(use_rot)
%     from=to+1;
%     to  =from+round(rot_ratio*batch_size)-1;
%     %get theta
%     theta = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);
%     %perform rotation-crop on a random sample
%     temp=imrotate(data(:,:,:,from:to),theta,'bilinear','crop');
%     %Resize to fit network input
%     if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
%         new_data(:,:,:,from:to)=get_resized_objects(temp,input_dims);
%     else
%         new_data(:,:,:,from:to)=temp;
%     end
%     %%get theta, set affine2d
%     %theta = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);
%     %tform = affine2d([cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1]);
%     %
%     %%perform rotation on a random sample
%     %random_rot_idxs=randperm(length(paths),floor(rot_freq*length(paths)));
%     %rotated_objs=imwarp(data(:,:,:,random_rot_idxs),tform);
%     %
%     %%crop rotated images to remove blanks
%     %crop_h = abs(ceil(sind(theta)*object_dims(2)));
%     %crop_w = abs(ceil(sind(theta)*object_dims(1)));
%     %cropped_objs = rotated_objs(1+crop_h:end-crop_h,1+crop_w:end-crop_w,:,:);
%     %
%     %%resize cropped images to fit network input
%     %data(:,:,:,random_rot_idxs) = get_resized_objects(cropped_objs,input_dims);
% end
% 
% %resize what's been left to fit the network's input
% from=to+1;
% if(from<=batch_size)
%     if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
%         new_data(:,:,:,from:batch_size) = get_resized_objects(data(:,:,:,from:end),input_dims);
%     else
%         new_data(:,:,:,from:batch_size) = data(:,:,:,from:end);
%     end
% end


%% PROCESSING TRAINING BATCH METHODS
function output_data = get_random_segments(data,dims)
object_dims = size(data(:,:,1,1));
for i=size(data,4):-1:1
    rnd_height = uint32((object_dims(1)-dims(1))*rand());
    rnd_width  = uint32((object_dims(2)-dims(2))*rand());
    output_data(:,:,:,i) = data(rnd_height+1:rnd_height+dims(1),rnd_width+1:rnd_width+dims(2),:,i);
end
end

function output_data = get_resized_objects(data,dims)
output_data=imresize(data,dims,'bilinear','antialiasing',false);
end


function output_data = get_projections(data,dims,rot_angle_bounds)

theta = rot_angle_bounds(1) + (rot_angle_bounds(2)-rot_angle_bounds(1))*rand(1,1);

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