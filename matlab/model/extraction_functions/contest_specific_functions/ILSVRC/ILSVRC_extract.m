function objs_fpaths = ILSVRC_extract(this_meta,dims,labels,appendTo)
%ILSVRC_EXTRACT This function is extracting the objects from the images
%using the metadata extracted from a previous step.
%% Explanation
%   It is taking as input 1 entry of metadata array and extracts the
%   objects to 'appendTo' directory. appendTo is set to be inside
%   cache/objects/'SET'
%   The dimensions of the object are set under 'dims' input as a 2-element
%   array [height width];
%   labels is a struct with all the semantics that should follow the
%   object. It is extensively described under 
%   'object_extraction/object_labels.m'
%   
%% AUTHOR: PROVOS ALEXIS
%  DATE:   20/5/2015
%  FOR:    VISION TEAM - AUTH

%% Image Handler
im.path     = this_meta.imdb_cor;
im.size     = this_meta.size;
im.data     = [];

%% object vars
[~,fname,~] = fileparts(im.path);
objs        = this_meta.objs;

%% Init 
counter=1;
objs_fpaths={};

%% Extract the objects and return the directories
for i=1:length(objs)
    obj = objs(i);
    filename   = strcat(fname,'_',num2str(i),'.mat');
    obj_fpath = fullfile(appendTo,filename);
	try
        if ~exist(obj_fpath,'file')
            im.data = read_image(im);
            xmin = obj.bndbox.xmin;
            xmax = obj.bndbox.xmax;
            ymin = obj.bndbox.ymin;
            ymax = obj.bndbox.ymax;
            if xmin==0
                xmin=1;
            end
            if ymin==0
                ymin=1;
            end
            %CREATE a temporary object
            obj_temp             = im.data((ymin:ymax),(xmin:xmax),:);
            % We turn off antialiasing to better match OpenCV's bilinear (SPP) 
            % Should we release use it, since we are performing
            % transformations on the fly?
            % Shouldnt we randomize the interpolation method on the fly?
            object.data          = imresize(obj_temp,dims, 'bilinear', 'antialiasing', false);
            %object.labels        = labels; %<<--- for debug
            object.uid           = labels.uid;
            %object.source        = im.path;%<<--- for debug
            save(obj_fpath,'object','-v6'); %<<--- remove v6 to save some space
        end
        objs_fpaths{counter}=obj_fpath;            
        counter = counter + 1;        
	catch err
        [~,name,~] = fileparts(im.path);
        APP_LOG('warning','Failed extraction for image %s',name);
        APP_LOG('warning','Error message: %s',err.message);
	end
end
end


function image = read_image( im )
%READ_IMAGE get the image handler. If the image is not loaded try to load
%it
if isempty(im.data)
    try
        im_temp = imread(im.path);
        if size(im_temp,3)==1
            im_temp = repmat(im_temp,1,1,3);
        end
        %im_temp = whitebalance(im_temp);
        image = imresize(im_temp,im.size, 'bilinear', 'antialiasing', false);
    catch err
        throw(err);
    end
else
    image = im.data;
end

end