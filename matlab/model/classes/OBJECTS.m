classdef OBJECTS < handle
    %OBJECTS This class is used to create/store/index the objects extracted
    %from the images using the metadata arrays that were extracted from 
    %METADATA class.
    %
    %   Properties:
    %               dims: The dimensions of the the extracted objects
    %               data: The filepaths of the extracted objects organised
    %                     per set/class for better visual inspection.
    %   Functions:
    %               build_objects: Is extracting the objects and building
    %                              the indexer who will be stored under
    %                              data in order to feed him to the net.
    %                save_objects: Is saving the indexer under cache
    %                              directory.
    %                load_objects: Is loading the indexer from the
    %                              specified cache directory that the model
    %                              was initialized. May throw error if file
    %                              is not present.
    %   AUTHOR: PROVOS ALEXIS
    %   DATE:   19/5/2015
    %   FOR:    vision team - AUTH
    
    properties
        dims;
        data;
    end
    methods
        %% INIT
        function obj = OBJECTS()
            obj.dims=[];
            obj.data=[];
        end
        %% SETTERS
        function obj = set_dims(obj,dims)
            obj.dims = dims;
        end
    end
    methods (Hidden = true)
        %% BUILD OBJECTS
        function obj = build_objects(obj,set,meta,paths,contest)
            obj.data = build_objs(set,meta,paths,obj.dims,contest);
        end
        %% SAVE OBJECTS
        function save_objects(obj,obj_file)
            APP_LOG('info',0,'Save Objects filepaths indexer...');
            data = obj.data;
            save(obj_file,'data','-v6');
            APP_LOG('info',0,'Objects filepaths indexer saved succesfully!');            
        end        
        %% LOAD OBJECTS
        function obj = load_objects(obj,obj_file)
            APP_LOG('header',0,'Loading Objects filepaths indexer from %s',obj_file);
            load(obj_file);
            obj.data = data;
            APP_LOG('info',0,'Objects filepaths indexer loaded succesfully!');
        end
    end
    
end

