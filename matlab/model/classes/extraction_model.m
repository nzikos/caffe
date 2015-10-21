classdef extraction_model < handle
    %extraction_model is used to obtain paths and extract metadata-objects 
    %from a specific contest. In case of a non supported contest, 
    %one can create his/her own functions and pass them to the function 
    %handlers under CONTEST object.

    %PROPERTIES short description:
    %   sets    : Contains the names of the datasets e.g.
    %             train,val,training,validation etc. Those sets are also the 
    %             names of the IMDB/metadata subfolders.
    %
    %   paths   : Paths of the metadata/IMDB/Cache.
    %
    %   contest : Contest specific functions.
    %   
    %   metadata: Class containing metadata manipulation functions and data.
    %
    %   objects : Class containing extracted objects from IMDB manipulation
    %             functions and data.
    %   LOGS    : Class containing Logs manipulation functions.
    %
    %AUTHOR: PROVOS ALEXIS
    %DATE:   19/5/2015
    %FOR:    vision team - AUTH
    
    properties
        sets;
        paths;
        contest;
        metadata;
        objects;
    end
    
    methods
%% CONSTRUCTOR
        function model = extraction_model()
            APP_LOG('enable',fullfile(pwd,'LOGS',strcat('Logs_',datestr(now, 'DD_mm_YYYY_HH_MM_SS'),'.txt')));
            model.sets=[];
            model.paths=[];
            model.contest=[];
            model.metadata=METADATA();
            model.objects =OBJECTS();
        end
%% SETTERS
        function set_sets(model,my_sets)
            model.sets     = SETS(my_sets);
        end
        function set_paths(model,meta,imdb,cache)
            model.paths    = PATHS(model.sets,meta,imdb,cache);            
        end
        function set_contest(model,contest)
            model.contest  = CONTEST(contest);            
        end
%% METADATA HANDLERS
        function print_metadata(model)
            model.metadata.print(model.sets.set,model.contest.class_names_map);
        end        
%% OBJECTS HANDLERS
        function load_objects(model)
            try
            	model.objects.load_objects(model.paths.objects_file);
            catch err
                APP_LOG('warning','%s',err.message);
                try
                    model.metadata.load_metadata(model.paths.meta_file);
                catch err
                    APP_LOG('warning','%s',err.message);
                    model.metadata.build_metadata(model.paths,model.contest,model.sets.set);
                    model.metadata.check(model.sets.set);
                    model.metadata.save_metadata(model.paths.meta_file);
                end
                model.objects.build_objects(model.sets.set,model.metadata,model.paths,model.contest);
                model.objects.save_objects(model.paths.objects_file);
            end
        end
    end
end