function  output_data  = get_random_rotations(theta_bounds,data,dims)
%GET_RANDOM_ROTATIONS Random rotations of objects
%   1. Rotates a batch of objects by a random angle.
%   2. Crops objects to remove blanks according to crop_blanks implementation.
%   3. Resizes objects to fit dims.
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

    theta       = theta_bounds(1) + (theta_bounds(2)-theta_bounds(1))*rand(1,1);
    
    R           = [cosd(theta) -sind(theta) 0;
                   +sind(theta) cosd(theta) 0;
                   0            0           1];
               
    tform       = affine2d(R);
    temp_data   = imwarp(data,tform);
    
    temp_data   = crop_blanks(temp_data,size(data),tform);
    
    object_dims = size(temp_data);
    
    if object_dims(1)~=dims(1) || object_dims(2)~=dims(2)
        output_data = get_resized_objects(temp_data,dims);
    else
        output_data = temp_data;
    end
end

