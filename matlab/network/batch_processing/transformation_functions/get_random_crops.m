function output_data = get_random_crops( data,dims )
%GET_RANDOM_CROPS crop a random segment
%   In order to achieve translation invariance on CNNs a random crop is
%   applied on the objects data, resulting in a smaller segment.
%   It is the caller's responsibility to check that object_dims>dims
%   elsewhere this function will return him uncropped objects.
%   
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

object_dims = size(data(:,:,1,1));
for i=size(data,4):-1:1
    rnd_height           = uint32((object_dims(1)-dims(1))*rand());
    rnd_width            = uint32((object_dims(2)-dims(2))*rand());
    output_data(:,:,:,i) = data(rnd_height+1:rnd_height+dims(1),rnd_width+1:rnd_width+dims(2),:,i);
end
end