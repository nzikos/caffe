classdef BATCH_FACTORY < handle
    %BATCH_FACTORY class purpose is to create a batch of images in order to
    %be processed by caffe.
    %% PROPERTIES
    %	    net_structure          : network's structure model
    %	    transforms 		   : samples transformations model
    %	    input_norm	           : input normalization method
    %
    %       train_objects          : pool of objects scheduled for training
    %       validation_objects     : pool of objects scheduled for validation
    %       validation_objects_pos : index for validation_objects
    %   
    %       train_queue            : queue for async batch
    %                                load/augmentation during training
    %       validation_queue       : queue for async batch
    %                                load during validation
    %       train_queue_size       : size of queue
    %       validation_queue_size  : size of queue
    %
    %       train_queue_idx        : index for training queue
    %       validation_queue_idx   : index for validation queue
    %
    %       train_curr_paths       : Cell{train_queue_size} contains paths
    %                                currently loaded on each batch of the queue
    %
    %% SETTERS
    %       set_async_queue_size   : Sets the size of train and validation
    %                                queues and starts matlab's pool
    %
    %       set_train_objects      : Sets the directories with the training 
    %                                samples along with their class
    %                                frequencies as passed from network.m
    %                                and extracted from extraction_model.m
    %
    %       set_validation_objects : Sets the directories with the
    %                                validation samples as passed from
    %                                network.m and extracted from
    %                                extraction_model.m
    %
    %       set_mean_std           : Sets the mean and std as extracted
    %                                from training set, inside INPUT_NORM
    %
    %       set_norm_type          : Sets input normalization type
    %
    %       set_epsilon            : Sets epsilon var in variance
    %                                normalization (default check INPUT_NORM)
    %
    %       set_sampling_method    : Sets whether it'll be used sampling with 
    %                                replacement or not while sampling
    %                                the classes which will be contained
    %                                inside a batch. Also sets the prior
    %                                probabilities of each class to be seen
    %                                inside a batch.
    %       set_tform              : Sets the type of transformations who
    %                                will be applied in a batch. (See
    %                                TRANSFORMS)
    %       
    %% METHODS
    %       assign_training_paths  : Implements the sampling method through
    %                                the training set.
    %
    %       assign_validation_work : Loads the validation samples paths into
    %                                async workers.
    %
    %       create_training_batch  : Fetches loaded and transformed samples
    %                                of training set from async worker (FIFO QUEUE).
    %                                Loads the sampled training samples
    %                                paths into "the same" async
    %                                worker.(Same id)
    %
    %       create_test_batch      : Takes a batch of images during the
    %                                testing / operational phase in HxWxD.
    %                                Permutes it to WxHxD to comply with
    %                                Caffe, and applies the specified input
    %                                normalization.
    %                                   
    %       prepare_train_batch      : Used to initialize the training 
    %                                  FIFO QUEUE.
    %
    %       async_load_current_batch : create_trainin_batch's sub-function
    %
    %       prepare_validation_batch : Used to initialize the validation 
    %                                  FIFO QUEUE. Check inside for TODO
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   20/5/2015
    %       FOR:    VISION TEAM - AUTH
    properties  
        net_structure;

        input_norm       = INPUT_NORM();
        transforms       = TRANSFORMS();
                
        train_objects          = [];
        train_objects_pool     = [];
        validation_objects     = [];
        validation_objects_pos = 0;

        sampling_frequencies   = [];
        sampling_with_replacement;

        %Training batch queue
        train_queue;
        validation_queue;        
        
        train_queue_size = 1;
        validation_queue_size = 1;
        
        train_queue_idx  = 1;
        validation_queue_idx =1;
        
        train_curr_paths;
    end
    
    methods
        %% CONSTRUCTOR
        function obj = BATCH_FACTORY(net_structure)
            obj.net_structure = net_structure;
        end

        %% SETTERS
        %SET QUEUE SIZE
        function set_async_queue_size(obj,size)
            obj.train_queue_size = size;
            obj.validation_queue_size = size;
            if(~isempty(gcp('nocreate')))
                p=gcp();
                if(p.NumWorkers~=obj.train_queue_size)
                    delete(gcp());
                    parpool(obj.train_queue_size);
                end
            else
                parpool(obj.train_queue_size);
            end                        
        end
                
        %SET training objects paths
        function set_train_objects(obj,arg_objects)
            for i=length(arg_objects):-1:1
                obj.train_objects{i,1}      = arg_objects(i).paths;
            end
            obj.train_objects_pool = obj.train_objects;
            obj.sampling_frequencies = ones(length(obj.train_objects_pool))/length(obj.train_objects_pool);
        end
        
        %SET validation objects paths
        function set_validation_objects(obj,arg_objects)
            obj.validation_objects       = vectorize_objects_fpaths(arg_objects);
        end
        
        %Internal SET mean and std on batch during training
        function set_mean_std(obj,arg)
            obj.input_norm.set_mean_std(arg);
        end
        
        %SET Whether to use mean/std for batch normalization
        function set_norm_type(obj,arg_type)
            obj.input_norm.set_norm_type(arg_type,obj.net_structure.objects_size);
        end
        
        %SET Epsilon in variance normalization
        function set_epsilon(obj,arg_type)
            obj.input_norm.set_epsilon(arg_type);
        end
        
        %SET Sampling frequencies per category
        function set_sampling_method(obj,w_replacement,frequencies)
            if length(frequencies)~=length(obj.sampling_frequencies)
                APP_LOG('last_error','sampling frequencies are unequal with training set classes');
            end
            obj.sampling_frequencies      = frequencies;
            obj.sampling_with_replacement = w_replacement;
        end
        
        %SET Transforms
        function set_tform(obj,type,args)
            obj.transforms.set_tform(type,args);
        end
        
        %% METHODS
        
        %% Paths assignment to workers
        function assign_training_paths(obj,which_queue_id)
            if obj.sampling_with_replacement
                class_idxs = randsample(length(obj.train_objects),obj.net_structure.train_batch_size,true,obj.sampling_frequencies);
            else
                class_idxs = randsampleWR(length(obj.train_objects),obj.net_structure.train_batch_size,obj.sampling_frequencies);
            end
            
            unique_class_idxs = unique(class_idxs);
            for i=length(unique_class_idxs):-1:1
                how_many_samples(i) = sum(class_idxs==unique_class_idxs(i));
            end
            
            c=1; % fetched samples counter
            for i=length(unique_class_idxs):-1:1
                class_idx          = unique_class_idxs(i);
                how_many_i_need    = how_many_samples(i);
                how_many_are_there = length(obj.train_objects_pool{class_idx});
                
                if how_many_i_need<=how_many_are_there
                    local_array(c:c+how_many_i_need-1,1)=obj.train_objects_pool{class_idx}(end-how_many_i_need+1:end);
                    c=c+how_many_i_need;
                    obj.train_objects_pool{class_idx}(end-how_many_i_need+1:end)=[]; %remove used samples
                else
                    while how_many_i_need>how_many_are_there
                        local_array(c:c+how_many_are_there-1,1)=obj.train_objects_pool{class_idx};
                        c=c+how_many_are_there;
                        how_many_i_need=how_many_i_need - how_many_are_there;
                        %refill
                        num = length(obj.train_objects{class_idx});
                        rand_idxs  = randperm(num,num);
                        obj.train_objects_pool{class_idx}=obj.train_objects{class_idx}(rand_idxs);                        
                        
                        how_many_are_there = length(obj.train_objects_pool{class_idx});
                    end
                    if how_many_i_need>0
                        local_array(c:c+how_many_i_need-1,1)=obj.train_objects_pool{class_idx}(end-how_many_i_need+1:end);
                        c=c+how_many_i_need;
                        obj.train_objects_pool{class_idx}(end-how_many_i_need+1:end)=[]; %remove used samples
                    end                    
                end
            end
            
            obj.train_curr_paths{which_queue_id}=local_array;
        end
        
        function assign_validation_work(obj,which_queue_id)
            from = obj.validation_objects_pos;
            to   = obj.validation_objects_pos + obj.net_structure.val_batch_size;
            if(to <= length(obj.validation_objects))
                obj.validation_objects_pos = to;                
                x.paths                    = obj.validation_objects(from+1:to);
                obj.validation_queue{which_queue_id}   = parfeval(@async_read_from_disk,1,x); %put some work
            else
                obj.validation_queue{which_queue_id}   = []; %no work assigned
            end
        end        
        
        %% Batch creation methods
        function out = create_training_batch(obj)          
            out=fetchOutputs(obj.train_queue{obj.train_queue_idx});%fetch old work

            out{1,1}=obj.input_norm.do_normalization(out{1,1});
            out{2,1}=obj.net_structure.layers{end}.transform_labels(out{2,1});
            
            obj.assign_training_paths(obj.train_queue_idx);%assign new paths            
            obj.async_load_current_train_batch(obj.train_queue_idx); %assign new work
            
            obj.train_queue_idx=obj.train_queue_idx+1;
            if(obj.train_queue_idx>obj.train_queue_size)
                obj.train_queue_idx=1;
            end
        end
        
        function out = create_validation_batch(obj)    
            if(~isempty(obj.validation_queue{obj.validation_queue_idx}))
                out=fetchOutputs(obj.validation_queue{obj.validation_queue_idx});
                out{1,1}=obj.input_norm.do_normalization(out{1,1});
                out{2,1}=obj.net_structure.layers{end}.transform_labels(out{2,1});
            else
                out=[]; %stop signal for VALIDATION.m
                return;
            end
            obj.assign_validation_work(obj.validation_queue_idx);
            
            obj.validation_queue_idx=obj.validation_queue_idx+1;
            if(obj.validation_queue_idx>obj.validation_queue_size)
                obj.validation_queue_idx=1;
            end
        end
        
        function out = create_test_batch(obj,data)
            out{1,1} = permute(data, [2 1 3 4]);
            out{1,1} = obj.input_norm.do_normalization(out{1,1});
        end
        
        %% Prepare asynchronous Batch creation
        function prepare_train_batch(obj) 
            %We are assuming that batch size is smaller than number of objects 
            for i=1:obj.train_queue_size;
                obj.assign_training_paths(i);
                obj.async_load_current_train_batch(i);
            end
        end
        
        function async_load_current_train_batch(obj,i)
            x.paths             = obj.train_curr_paths{i};
            x.input_dims.height = obj.net_structure.objects_size(1);
            x.input_dims.width  = obj.net_structure.objects_size(2);
            x.transforms        = obj.transforms;
            obj.train_queue{i}  = parfeval(@async_read_from_disk_and_preprocess,1,x);
%           out = async_read_from_disk_and_preprocess(x);
        end

        function prepare_validation_batch(obj)            
            %% TODO
            %Initial assumption - we are assuming that validation_batch_size is less than total validation objects.
            %Last samples that are less than batch_size are not being used.
            %Needs changes...            
            obj.validation_objects_pos = 0;
            for i=1:obj.validation_queue_size
                obj.assign_validation_work(i);
            end
        end
    end    
end

