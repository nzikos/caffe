classdef PATHS
    %PATHS used to store and handle filepaths and directories
    %
    %   This class is part of extraction_model()
    %
    %%  PROPERTIES
    %
    %       meta         : Struct with train/val directories which contain
    %                      the samples metadata.
    %
    %       imdb         : Struct with train/val directories which contain
    %                      the samples data.
    %
    %       objects      : Struct with train/val directories which contain
    %                      the extracted samples.
    %
    %       cache        : directory of cache
    %
    %       meta_file    : Absolute filepath to metadata file
    %
    %       objects_file : Absolute filepath to objects file
    %
    %       sets         : the naming of datasets (e.g. training,
    %                      validation,train,val,etc)
    %
    %%   METHODS
    %
    %       print_paths  : Used to print paths in console
    %
    %%  AUTHOR: PROVOS ALEXIS
    %   DATE:   19/5/2015
    %   FOR:    vision team - AUTH    
    properties
        meta
        imdb
        objects
        cache
        meta_file
        objects_file
        sets
    end
    
    methods (Hidden = true)
        function obj = PATHS(sets,meta,imdb,cache)
            if strcmp(class(sets),'SETS')
                %obj.sets = SETS(sets.set);
                obj.sets = sets;
                for i=1:length(sets.set)
                    obj.meta.(sets.set{i})=[];
                    obj.imdb.(sets.set{i})=[];
                    obj.objects.(sets.set{i})=[];
                end
                %SET METADATA PATHS
                for i=1:length(sets.set)
                        dirc = fullfile(meta,sets.set{i});
                        obj.meta.(sets.set{i})=dirc;
                        handle_dir(dirc,'throw error');
                end
                %SET IMAGE PATHS
                for i=1:length(sets.set)
                        dirc = fullfile(imdb,sets.set{i});
                        obj.imdb.(sets.set{i})=dirc;
                        handle_dir(dirc,'throw error');
                end
                %SET CACHE / meta-object FILES
                dirc = cache;
                handle_dir(dirc,'create');
                obj.cache = dirc;
                for i=1:length(sets.set)
                        dirc = fullfile(cache,'objects',sets.set{i});
                        obj.objects.(sets.set{i})=dirc;
                        handle_dir(dirc,'create');
                end
                obj.meta_file = fullfile(cache,'meta.mat');
                obj.objects_file  = fullfile(cache,'objects.mat');
            end
        end
        function obj = print_paths(obj)
            APP_LOG('header','PATHS INFO');
            APP_LOG('info','Cache directory: %s',obj.cache);
            for i=1:length(obj.sets.set)
                APP_LOG('info','Meta %s set dir: %s',obj.sets.set{i},...
                                             obj.meta.(obj.sets.set{i}));
                APP_LOG('info','Imdb %s set dir: %s',obj.sets.set{i},...
                                             obj.imdb.(obj.sets.set{i}));
                APP_LOG('info','Objects %s set dir: %s',obj.sets.set{i},...
                                             obj.objects.(obj.sets.set{i}));
            end
            APP_LOG('info','Metadata file: %s',obj.meta_file);
            APP_LOG('info','Objects Index file: %s',obj.objects_file);
        end
    end
    
end

