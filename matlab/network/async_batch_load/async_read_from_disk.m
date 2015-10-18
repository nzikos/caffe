function out = async_read_from_disk(x)
%ASYNC_READ_FROM_DISK This function is used by batch_factory in order to
%load asynchronously the batch from the hard disk. 

% paths   : The paths of the objects that'll be loaded as a Cell Array of
%           strings.
% data    : The objects/images as HxWx3xN (where N is the batch size)
% uids    : The unique ids of the class as defined from the extraction_model
%           matrix with dimensions of 1x1x1xN
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH
%% DO NOT DELETE THIS LINE
java.lang.System.gc(); %---MEMORY LEAK SOLVED!!!--- Why matlab, why...

paths = x.paths;
for i=length(paths):-1:1
    tmp = load(paths{i});
    out{1,1}(:,:,:,i)=tmp.object.data;
    out{2,1}(1,1,1,i)=tmp.object.uid;
end
end

