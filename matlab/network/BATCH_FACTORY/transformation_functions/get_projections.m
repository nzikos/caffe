function output_data = get_projections( data,dims )
%GET_PROJECTIONS Random projections of objects
%   1. Rotates a batch of objects by a random angle and performs a random projection.
%   2. Crops objects to remove blanks according to crop_blanks implementation.
%   3. Resizes objects to fit dims.
%
%   This function is under testing. Parameters need to be carefully
%   assigned in order to preserve the label after the transformation
%
%
%   This function is part of TRANSFORMS() Class.
%
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

theta = -20 + 40*rand(1,1);
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
tmp1=crop_blanks(tmp,size(data),tform);

output_data = get_resized_objects(tmp1,dims);
clear tmp1;
end

