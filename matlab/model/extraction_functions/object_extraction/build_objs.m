function data = build_objs(set,meta,paths,dims,contest)
%BUILD_OBJS This function is used to extract the objects from the images
%using the previously build metadata.

%%   Detailed explanation
%    This function is in charge to create the objects filepaths indexers
%    for all datasets. It is applying the same function to all datasets.
%    It is mainly working on two loops. The first one is iterating through
%    the objects of each class. The nested one is iterating through the
%    datasets. In the end of every class it is printing:
%       1. The uid in integer format of the class. 
%       2. The class name in human understandble format. 
%       3. The class id from the contest.
%       4. The number of objects extracted for each class.
%%  Variables explanation
%       class_names_map : Containers.map including a 1-1 relation model of
%                         contest id class -> human understanable class name.
%        extract_object : The handler who will call the function which will
%                         extract the object and return the filepaths.
%               classes : The classes.
%                labels : The semantics which will follow a specific object
%                         throughout his lifetime. Check object_labels.m
%                         for more information.
%              appendTo : The directory in which the objects of the j-th 
%                         iteration will be extracted.
%         shrinked_meta : The metadata for a specific class. All the
%                         objects from an entry that belong to different
%                         class have been removed.
%        par_obj_fpaths : The objects filepaths organised in a cell of
%                         cells array. Each subcell is containing multiple
%                         object filepaths extracted from an image.
%        ser_obj_fpaths : The previous array as a cell array.
%%   AUTHOR: PROVOS ALEXIS
%    DATE:   20/5/2015
%    FOR:    VISION TEAM - AUTH

%%  PRINT HEADER
APP_LOG('header','Extracting objects...');
out = sprintf('%4s%22s[%9s]','N','Name','classID');
for i=1:length(set)
    out = [out sprintf('%10s',set{i})];
end
APP_LOG('header','%s',out);

%% Init
class_names_map = contest.class_names_map;
extract_object  = contest.extract_objects;

classes = meta.rmap.(set{1}).keys; %SAME FOR ALL SETS (CHECKED BEFORE CALLING)
                                   %defining classes outside main loop 
                                   %guarantees correspondence inside 
                                   %objs_fpaths.
data=[];                                  
%% For each class
for j=1:length(classes)

    % Get the labels
    labels = object_labels(class_names_map,classes,j); 
    
    % Prepare class print output
    out = sprintf('%4s%22s[%9s]',num2str(j),class_names_map(classes{j}),classes{j});
    
    % For each dataset
	for i=1:length(set)

        %Extract Objects of this set/class to
        appendTo    = fullfile(paths.objects.(set{i}),labels.name);
        handle_dir(appendTo,'create');
        
        %Get only the objects that belong in j-th class of i-th set
        shrinked_meta = get_shrinked_meta(meta.data.(set{i}),...
                                          meta.rmap.(set{i}),...
                                          labels.contestID);     
                                      
        %Get dimensions of extracted object (dataset relative)
        this_dims = dims.(set{i});
        
        %EXTRACT OBJECTS
        par_objs_fpaths = {};
        parfor k=1:length(shrinked_meta)
            par_objs_fpaths{k} = feval(extract_object,shrinked_meta(k)...
                                                     ,this_dims...
                                                     ,labels...
                                                     ,appendTo...
                                                     );
        end
        
        %Serialize them
        counter=1;
        ser_objs_paths  = {};
        for k=1:length(shrinked_meta)
            for l=1:length(par_objs_fpaths{k})
                ser_objs_paths{counter,1}=par_objs_fpaths{k}{l};
                counter = counter +1;
            end
        end
        
        %Set on printing output the number of objects extracted for the i-th set
        out = [out sprintf('%10s',num2str(length(ser_objs_paths)))];
        
        %Append the paths
        data.(set{i})(j).labels= labels;
        data.(set{i})(j).paths = ser_objs_paths;
        
	end
    
    %Print class info
    APP_LOG('info','%s',out);
    
end
end

