function out = async_read_from_disk_and_preprocess(x)
%ASYNC_READ_FROM_DISK_AND_PREPROCESS  This function is used by batch_factory
% in order to load asynchronously the batch from the hard disk and process
% the objects before network feeds them to caffe.

%% ARGUMENTS
%   Argument x is a struct which contains:
%       paths      : The paths of the objects that'll be loaded as a
%                    Cell Array of strings.
%
%       input_dims : The dimensions of caffe input layer.
%
%       tforms     : Object of class TRANSFORMS which is used to carry on
%                    the transformations.
%
%% RETURNS
% data    : The objects/images as HxWx3xN (where N is the batch size)
% uids    : The unique ids of the class as defined from the extraction_model
%           matrix with dimensions of 1x1xKxN
%
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

%PREPARE ATTRIBUTES
paths        = x.paths;
input_dims   = [x.input_dims.width x.input_dims.height]; %Remember inconsistency between matlab and caffe dimensions
tforms       = x.transforms;

%PERFORM RANDOM PERMUTATION (used to randomize the way transforms are applied per sample)
paths=paths(randperm(length(paths),length(paths)));

%EXTRACT OBJECTS
for i=length(paths):-1:1
    tmp = load(paths{i});
    data(:,:,:,i)=tmp.object.data;
    uids(i,:)=tmp.object.uid;
end

%OUTPUT
out{1,1}=tforms.apply_tforms(data,input_dims);
out{2,1}=uint16(uids);

% DO NOT DELETE THIS LINE.
% Bug found under ubuntu 14.04 / Matlab R2015a
% Workers garbage collector is not cleaning efficiently the memory which
% causes extensive memory usage for workers making the system unresponsive
% while calling parfeval several times from within a loop.
java.lang.System.gc();
end
