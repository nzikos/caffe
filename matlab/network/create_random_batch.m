function [out,paths] = create_random_batch(paths,batch_size)
%CREATE_RANDOM_BATCH Load a random batch with data,vectors and uids 
%from hard disk to memory in order to be fed to caffe.

idxs=randperm(length(paths),batch_size);
this_paths=paths(idxs);
paths(idxs,:)=[];
out={};
for j=1:batch_size
    out = put_obj_on_batch(out,j,this_paths{j});
end
end

function out = put_obj_on_batch(out,j,path)
        tmp = load(path);
        out{1,1}(:,:,:,j)=single(tmp.object.data)/255;
        out{2,1}(1,1,:,j)=single(tmp.object.labels.vector);
        out{3,1}(1,1,1,j)=single(tmp.object.labels.uid);
end