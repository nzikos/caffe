function [data,rmap] = build_meta(paths,dataset,set)
%   BUILD_META.m This function is used to build the metadata tables and
%   class maps that later will be used to export the objects from images
%
%   Build meta is in charge of the following actions:
%
%   1. Reads metadata from metadata files and stores them in a matlab array
%      of structs.
%   2. Maps classes to objects.

%   User is able to import his/her own function which is in charge of
%   extracting the metadata from a specified a dataset.
%   This can be done by supplying the custom function at the handler under
%   dataset.readmeta
%   The project is waiting as output from this function an array of structs
%   Each struct must contain the metadata of a specific file in the 
%   following way:
%
%   [imdb_cor]  The filename of the image or anything else in the imdb set 
%               that this object is found. //Can be a struct, a pointer to a binary file.
%               Depends who will extract the object.
%
%   [size]      2 element array containing [height width] this image had
%               when bounding boxes were extracted.
%               This is needed by the extract objects function because when
%               the bounding boxes were extracted on some datasets the
%               image was scaled to these dimensions.
%
%   [objs]      The object. A struct containing all the objects that exist
%               under the specific image.
%               [name]      The object name / class id.
%               [bndbox]    The bounding box, inside which, this object is 
%                           found. The bndbox is a struct containing the 
%                           pixel values of:
%                                              [xmin] [xmax] [ymin] [ymax]
%
%   AUTHOR: PROVOS ALEXIS
%   DATE:   7/5/2015
%   FOR:    vision team - AUTH

%%  Build metadata
    APP_LOG('header','BUILDING META');
    
    meta_ext    =dataset.meta_extension;
    meta_paths  =paths.meta;
    imdb_ext    =dataset.im_extension;
    imdb_paths  =paths.imdb;
    
    readmeta_function = dataset.read_meta;
    class_location = {'objs','name'};
      
	for i=1:length(set)
        APP_LOG('info','Extracting %s set metadata',set{i});
        data.(set{i}) = extract_meta(meta_paths.(set{i}),meta_ext,...
                                     imdb_paths.(set{i}),imdb_ext,...
                                     readmeta_function);
        
        APP_LOG('info','Mapping %s set metadata',set{i});
        rmap.(set{i}) = containers.Map('KeyType','char','ValueType','any');
        rmap.(set{i}) = map_relations(class_location, data.(set{i}),rmap.(set{i}));
	end
end


function meta = extract_meta(meta_dir,meta_ext,imdb_dir,imdb_ext,get_meta_from)
%% EXTRACT_META Get all metadata files and extract their data
%
%   This function is searching through a directory and all the
%   subdirectories recursively, returning the filepaths and the relative 
%   filepath of all files with the specified extension.
%
%   After that, it is calling @read_meta, which is a dataset specific
%   function. @read_meta reads all metadata files that've been found and 
%   returns the results so that can be used to export the objects from the
%   images.
%
%   AUTHOR: PROVOS ALEXIS
%   DATE:   7/5/2015
%   FOR:    vision team - AUTH 

%% GET FILES
    APP_LOG('info','Searching for %s files in %s',meta_ext,meta_dir);
    list    =super_get_file_list(meta_dir,meta_ext);
    n_files =length(list);
    APP_LOG('info','Extracting from %d files...',n_files);
    
%% EXTRACT
    meta = get_meta_from(list,imdb_dir,imdb_ext);
end



function map = map_relations(ifields,data,map,curr_depth,idx)
%MAP_RELATIONS Creates a map containing keys as values stored under ifields
%              and as values of the map are the indices where the specific 
%              value was found under data.
%              In general it is a 1->many relationship model.
%              Function is running recursively and depth of recursion is
%              the length of ifields.
%              You must initialize the map before calling map_relations.
%
%   Example of map initialization
%               map = containers.Map('KeyType','char','ValueType','any');
%
%   INPUT
%   ifields :   the path inside the struct where VALUE is located given as
%               a cell array. e.g. {'object','name'}.
%
%   data    :   the array of structs.
%
%   map     :   Initialized containers.map with specific attributes
%
%   OUTPUT
%   map     :   Map takes as input the value it is assigned to map and as
%               output returns the indices where this value is found under
%               data. (If the depth is of order 2 every value returned has 2
%               Indices, if depth is order of 3, returned values are 3 
%               indices and so on. -COMMENTED OUT FUNCTIONALITY) May return
%               multiple outputs if the value is found multiple times.
%
%  AUTHOR: PROVOS ALEXIS 
%  DATE:   20/4/2015
%  FOR:    vision team - AUTH

%% INITIALIZE
    full_depth=length(ifields);
    if(nargin<4)
        curr_depth=0;
%       idxs=zeros(full_depth,1);
    end
    curr_depth=curr_depth+1;
%% FILL MAP / GO RECURSIVE
    field=ifields{curr_depth};
    for i=1:length(data)
        if curr_depth==1;
            idx=i;
        end
        if isfield(data,field)
            this=data(i).(field);
%          	idx(curr_depth)=i;
            if curr_depth==full_depth
                if map.isKey(this)
                    map(this) = [map(this) idx];
                else
                    map(this) = idx;
                end
            else
                %Recursion, go deeper
                map=map_relations(ifields,this,map,curr_depth,idx);
            end            
        end
    end
end


