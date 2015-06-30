%CREATE_RANDOM_BATCH_W_RND_SGMNT
% Load a random batch with data,vectors and uids from hard disk to 
% memory in order to be fed to caffe. Loaded Object's dimension must
% be greater than network input dims.
function [out,paths] = create_random_batch_w_rnd_sgmnt(paths,batch_size,use_flip,dims,stride)
	idxs=randperm(length(paths),batch_size/(1+use_flip));
	this_paths=paths(idxs);
	paths(idxs,:)=[];
	out={};

    %CALC RND SEGMENT ONCE PER BATCH
    tmp = load(this_paths{1});

    object_dims = size(tmp.object.data);
    rnd_height  = uint16((object_dims(1)-dims(1))*rand());
    rnd_height  = rnd_height - mod(rnd_height,stride);
        
    rnd_width   = uint16((object_dims(2)-dims(2))*rand());
    rnd_width   = rnd_width  - mod(rnd_width ,stride);

    img = tmp.object.data(rnd_height+1:rnd_height+dims(1),rnd_width+1:rnd_width+dims(2),:);

    out{1,1}(:,:,:,1)=single(img)/255;
    out{2,1}(1,1,1,1)=single(tmp.object.labels.uid);
    out{3,1}(1,1,:,1)=single(tmp.object.labels.vector);
    %------------------------------
    
	for j=2:batch_size/(1+use_flip)
        tmp = load(this_paths{j});

        img = tmp.object.data(rnd_height+1:rnd_height+dims(1),rnd_width+1:rnd_width+dims(2),:);

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