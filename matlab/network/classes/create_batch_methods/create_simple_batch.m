%CREATE_SIMPLE_BATCH
% Load a batch with data,vectors and uids from hard disk to memory 
% in order to be fed to caffe.
function out = create_simple_batch(paths,batch_size,dims)
	out={};
	for j=1:batch_size
        tmp = load(paths{j});
        img = imresize(tmp.object.data,dims,'bilinear','antialiasing',false);
        out{1,1}(:,:,:,j)=single(img)/255;
        out{2,1}(1,1,1,j)=single(tmp.object.labels.uid);
        out{3,1}(1,1,:,j)=single(tmp.object.labels.vector);
	end
end