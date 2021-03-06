function data_out = crop_blanks( data_in ,object_dims, tform )
%CROP_BLANKS Crop blank regions after an applied transformation
%            This function is used as a helper function to crop the most out of
%            blanks of samples which were previously transformed.
%
%   This function is part of TRANSFORMS() Class.
%
%% AUTHOR: PROVOS ALEXIS
%   DATE:   5/11/2015
%   FOR:    VISION TEAM - AUTH

u = [1              1;
     1              object_dims(1);
     object_dims(2) 1;
     object_dims(2) object_dims(1)];

v = tform.transformPointsForward(u);

v(:,1)=(v(:,1)-min(v(:,1)))+1;
v(:,2)=(v(:,2)-min(v(:,2)))+1;
v=ceil(v);

min_v_h = sort(v(:,1));
min_v_w = sort(v(:,2));

y_1 = floor((min_v_h(2)-min_v_h(1))/2) + 1;
y_2 = floor((min_v_h(4)-min_v_h(3))/2) + min_v_h(3) - 1;

x_1 = floor((min_v_w(2)-min_v_w(1))/2) + 1;
x_2 = floor((min_v_w(4)-min_v_w(3))/2) + min_v_w(3) - 1;

data_out = data_in(x_1:x_2,y_1:y_2,:,:);

end