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
        LOGS;
    end
    
    methods
%% CONSTRUCTOR
        function model = extraction_model()
            diary off;
            model.sets=[];
            model.paths=[];
            model.contest=[];
            model.LOGS=[];
            model.enable_logs();
            model.metadata=METADATA();
            model.objects =OBJECTS();
            diary on;
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
        function build_metadata(model)
            model.metadata = model.metadata.build_metadata(model.paths,model.contest,model.sets.set);
        end
        function check_metadata(model)
            model.metadata.check(model.sets.set);
        end
        function print_metadata(model)
            model.metadata.print(model.sets.set,model.contest.class_names_map);
        end
        function load_metadata(model)
            model.metadata = model.metadata.load_metadata(model.paths.meta_file);
        end
        function save_metadata(model)
            model.metadata.save_metadata(model.paths.meta_file);
        end
        
%% OBJECTS HANDLERS
        function build_objects(model)
            model.objects = model.objects.build_objects(model.sets.set,model.metadata,model.paths,model.contest);
        end
        function load_objects(model)
            model.objects = model.objects.load_objects(model.paths.objects_file);
        end
        function save_objects(model)
            model.objects.save_objects(model.paths.objects_file);
        end
    end
    
    methods (Hidden = true)
        %% FUNCTIONS
        function enable_logs(model)            
            model.LOGS = fullfile(pwd,'LOGS',strcat('Logs_',datestr(now, 'DD_mm_YYYY_HH_MM_SS'),'.txt'));
            [logs_dir,~,~]  =   fileparts(model.LOGS);
            if ~exist(logs_dir,'dir')
                try
                    mkdir(logs_dir);
                catch err
                    error(err.message);
                end
            end
            diary(model.LOGS);
        end
    end 
end