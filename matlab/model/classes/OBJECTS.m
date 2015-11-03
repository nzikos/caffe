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
        mean;
        std;
        get_mean_std = 0;
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
        function obj = compute_mean_std(obj,value)
            obj.get_mean_std = value;
        end
    end
    methods
        %% SAVE OBJECTS
        function save_objects(obj,obj_file)
            APP_LOG('info','Save Objects filepaths indexer...');
            this.data         = obj.data;
            this.dims         = obj.dims;
            this.mean         = obj.mean;
            this.std          = obj.std;
            this.get_mean_std = obj.get_mean_std;
            save(obj_file,'this','-v6');
            APP_LOG('info','Objects filepaths indexer saved succesfully!');
        end
        %% LOAD OBJECTS
        function load_objects(obj,obj_file,set)
            APP_LOG('header','Loading Objects filepaths indexer from %s',obj_file);
            load(obj_file);
            obj.data         = this.data;
            obj.dims         = this.dims;
            obj.mean         = this.mean;
            obj.std          = this.std;
            APP_LOG('info','Objects filepaths indexer loaded succesfully!');
            if obj.get_mean_std
                if ~this.get_mean_std
                    obj.mean_and_std_computation_function(set);
                    obj.save_objects(obj_file);
                end
            end
        end
        %% BUILD OBJECTS
        function build_objects(obj,set,meta,paths,dataset)
            obj.data = build_objs(set,meta,paths,obj.dims,dataset);
            obj.mean_and_std_computation_function(set);
        end
        
        %% COMPUTE MEAN STD
        function mean_and_std_computation_function(obj,set)
            if obj.get_mean_std
                APP_LOG('info','Extracting mean and standard deviation from training samples...');
                tmp_mean        = permute(zeros([obj.dims.(set{1}) 3]),[2 1 3]);
                variance        = permute(zeros([obj.dims.(set{1}) 3]),[2 1 3]);
                counter         = 0;

                for i = length(obj.data.(set{1})):-1:1 
                    temp = obj.data.(set{1})(i).paths;
                    parfor j = 1:length(temp)
                        tmp = load(temp{j});
                        tmp_mean= tmp_mean+double(tmp.object.data);
                        counter = counter +1;
                    end
                    APP_LOG('debug','%4s%22s[%9s] mean extraction done',num2str(i),obj.data.(set{1})(i).labels.name,obj.data.(set{1})(i).labels.dataset_ID);
                end
                tmp_mean = tmp_mean./counter;
                                
                for i = 1: length(obj.data.(set{1}))
                    temp = obj.data.(set{1})(i).paths;                    
                    parfor j = 1:length(temp)
                        tmp = load(temp{j});
                        variance = variance + (double(tmp.object.data) - tmp_mean).^2;
                    end
                    APP_LOG('debug','%4s%22s[%9s] std extraction done',num2str(i),obj.data.(set{1})(i).labels.name,obj.data.(set{1})(i).labels.dataset_ID);
                end               
                obj.std  = single(sqrt(variance./(counter-1))); %assuming unknown mean
                obj.mean = single(tmp_mean);
            end            
        end
    end
    
end

