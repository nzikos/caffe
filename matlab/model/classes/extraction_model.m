classdef extraction_model < handle
    %extraction_model is used to obtain paths and extract metadata-objects 
    %from a specified dataset. In case of a non supported dataset, 
    %one can create his/her own functions and pass them to the function 
    %handlers under DATASET object.
    %
    %%  PROPERTIES
    %
    %   sets    : Contains the names of the datasets e.g.
    %             train,val,training,validation etc. Those sets are also the 
    %             names of the IMDB/metadata subfolders.
    %
    %   paths   : Paths of the metadata/IMDB/Cache.
    %
    %   dataset : dataset specific functions.
    %   
    %   metadata: Class containing metadata manipulation functions and data.
    %
    %   objects : Class containing extracted objects from IMDB manipulation
    %             functions and data.
    %
    %%  SETTERS
    %
    %   set_sets         : Used to set the naming of datasets (e.g.
    %                      training,validation,train,val,etc)
    %
    %   set_paths        : Used to set the expected paths to find:
    %                      1. Objects data
    %                      2. Objects metadata
    %                      And the paths used for samples extraction
    %
    %   set_dataset      : Declares the dataset which will be used
    %                      check DATASET for supported datasets
    %      
    %   set_dims         : Sets the dimensions HxW of extracted training 
    %                      and validation samples.
    %
    %   compute_mean_std : Computes pixel-wise mean and std of training set
    %
    %%  METHODS
    %
    %   print_paths      : Prints the paths
    %
    %   print_metadata   : Prints metadata as 
    %                           1. uid 
    %                           2. name(dataset_id)
    %                           3. number of detected meta
    %
    %   load_objects     : Used to 1. Extract metadata
    %                              2. Extract samples
    %                              3. Extract mean std
    %                              4. Compute relative frequencies of
    %                                 classes.
    %
    %   get_class_frequencies : Returns the relative frequencies of samples
    %                           per class.
    % 
    %%  AUTHOR: PROVOS ALEXIS
    %   DATE:   19/5/2015
    %   FOR:    vision team - AUTH
    
    properties
        sets;
        paths;
        dataset;
        metadata;
        objects;
    end
    
    methods
%% CONSTRUCTOR
        function model = extraction_model()
            APP_LOG('enable',fullfile(pwd,'LOGS',strcat('Logs_',datestr(now, 'DD_mm_YYYY_HH_MM_SS'),'.txt')));
            model.sets=[];
            model.paths=[];
            model.dataset=[];
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
        function set_dataset(model,arg_dataset)
            model.dataset  = DATASETS(arg_dataset);            
        end
        function set_dims(model,arg_dims)
            model.objects.set_dims(arg_dims);
        end
        function compute_mean_std(model,arg_val)
            model.objects.compute_mean_std(arg_val);
        end
%% PATHS HANDLERS
        function print_paths(model)
            model.paths.print_paths();
        end        
%% METADATA HANDLERS
        function print_metadata(model)
            model.metadata.print(model.sets.set,model.dataset.class_names_map);
        end        
%% OBJECTS HANDLERS
        function load_objects(model)
            try
            	model.objects.load_objects(model.paths.objects_file,model.sets.set);
            catch err
                APP_LOG('warning','%s',err.message);
                try
                    model.metadata.load_metadata(model.paths.meta_file);
                catch err
                    APP_LOG('warning','%s',err.message);
                    model.metadata.build_metadata(model.paths,model.dataset,model.sets.set);
                    model.metadata.check(model.sets.set);
                    model.metadata.save_metadata(model.paths.meta_file);
                end
                model.objects.build_objects(model.sets.set,model.metadata,model.paths,model.dataset);
                model.objects.save_objects(model.paths.objects_file);
            end
        end
        function freqs = get_class_frequencies(model,arg_set)
            switch arg_set
                case 'train'
                    idx= 1;
                case 'validation'
                    idx = 2;
                otherwise
                    APP_LOG('last_error','Unknown set. Use train/validation');
            end
            freqs = model.objects.class_frequencies{idx};
        end
    end
end
