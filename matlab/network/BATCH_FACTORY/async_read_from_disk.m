function out = async_read_from_disk(x)
%ASYNC_READ_FROM_DISK This function is used by batch_factory in order to
%load asynchronously the batch from the hard disk. 

% paths   : The paths of the objects that'll be loaded as a Cell Array of
%           strings.
% data    : The objects/images as HxWx3xN (where N is the batch size)
% uids    : The unique ids / ground truth labels of the class as defined 
%           from the extraction_model. Output layer's sub-functions expects
%           them in NxK format in order to transform them to CAFFE's 
%           expected format.
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

paths = x.paths;
for i=length(paths):-1:1
    tmp = load(paths{i});
    data(:,:,:,i)=tmp.object.data;
    uids(i,:)=tmp.object.uid;
end
out{1,1} = data;
out{2,1} = uint16(uids);
%% DO NOT DELETE THIS LINE
java.lang.System.gc();
end

