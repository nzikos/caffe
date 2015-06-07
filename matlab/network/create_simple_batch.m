function out = create_simple_batch(paths,batch_size)
%CREATE_SIMPLE_BATCH Load a batch with data,vectors and uids 
%from hard disk to memory in order to be fed to caffe.

out={};
for j=1:batch_size
    out = put_obj_on_batch(out,j,paths{j});
end
end

function out = put_obj_on_batch(out,j,path)
        tmp = load(path);
        out{1,1}(:,:,:,j)=single(tmp.object.data)/255;
        out{2,1}(1,1,1,j)=single(tmp.object.labels.uid);
        out{3,1}(1,1,:,j)=single(tmp.object.labels.vector);        
end