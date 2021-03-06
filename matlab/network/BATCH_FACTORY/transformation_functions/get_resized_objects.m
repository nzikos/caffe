function output_data = get_resized_objects(data,dims)
%GET_RESIZED_OBJECTS Resize objects to fit CNN input
%   bilinear and antializing false attributes inhererited from SPP-net
%
%   This function is part of TRANSFORMS() Class.
%
output_data=imresize(data,dims,'bilinear','antialiasing',false);
end

