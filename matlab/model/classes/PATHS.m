classdef PATHS
    %PATHS Summary of this class goes here
    %   Detailed explanation goes here
    
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

