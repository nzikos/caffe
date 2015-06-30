%CREATE_RANDOM_BATCH 
% Load a random batch with data,vectors and uids 
% from hard disk to memory in order to be fed to caffe.        
function [out,paths] = create_random_batch(paths,batch_size,use_flip,dims)   
    idxs=randperm(length(paths),batch_size/(1+use_flip));
    this_paths=paths(idxs);
    paths(idxs,:)=[];
    out={};
    for j=1:(batch_size/(1+use_flip))
        tmp = load(this_paths{j});
        img = imresize(tmp.object.data,dims,'bilinear','antialiasing',false);
        out{1,1}(:,:,:,j)=single(img)/255;
        out{2,1}(1,1,1,j)=single(tmp.object.labels.uid);
        out{3,1}(1,1,:,j)=single(tmp.object.labels.vector); 
    end
    if use_flip
        for j=(batch_size/2)+1:batch_size
            out{1,1}(:,:,:,j)=flip(out{1,1}(:,:,:,j-batch_size/2),2);
            out{2,1}(1,1,1,j)=out{2,1}(1,1,1,j-batch_size/2);
            out{3,1}(1,1,:,j)=out{3,1}(1,1,:,j-batch_size/2);
        end
    end
end