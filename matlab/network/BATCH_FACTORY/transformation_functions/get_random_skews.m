function  output_data  = get_random_skews(bounds,data,dims)
%GET_RANDOM_ROTATIONS Random skews of objects
%   1. Performs skewing on a batch of objects by random angles.
%   2. Crops objects to remove blanks according to crop_blanks implementation.
%   3. Resizes objects to fit dims.
%
%   This function is part of TRANSFORMS() Class.
%
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

    if rand(1,1)>0.5
        shear_h       = bounds(1) + (bounds(2)-bounds(1))*rand(1,1);
        shear_v       = 0;
    else
        shear_v       = bounds(1) + (bounds(2)-bounds(1))*rand(1,1);
        shear_h       = 0;        
    end
    
    S           = [1       shear_v 0;
                   shear_h 1       0;
                   0       0       1];
               
    tform       = affine2d(S);
    temp_data   = imwarp(data,tform);
    
    temp_data   = crop_blanks(temp_data,size(data),tform);
    
    object_dims = size(temp_data);
    
    if object_dims(1)~=dims(1) || object_dims(2)~=dims(2)
        output_data = get_resized_objects(temp_data,dims);
    else
        output_data = temp_data;
    end
end

