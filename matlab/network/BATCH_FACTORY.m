classdef BATCH_FACTORY < handle
    %BATCH_FACTORY class purpose is to create a batch of images in order to
    %be processed by caffe.
    %% ATTRIBUTES
    %       use_mean_std           : Use mean and std extracted from
    %                                training set in order to normalize
    %                                samples. mean and std dims are [HxWxD]
    %       crop_freq              : Ratio of objects random cropping per batch.
    %       rot_freq               : Ratio of objects rotation per batch.
    %       skew_freq              : Ratio of skewed objects per batch.
    %       projections_freq       : Ratio of projected objects per batch.
    %       
    %       rot_theta_bounds       : [minimum_theta maximum_theta] in degrees.
    %       skew_bounds      : [minimum_theta maximum_theta] in degrees.
    %
    %       use_flipped            : use horizontally flipped objects or not (0/1).
    %
    %
    %       train_objects          : pool of objects scheduled for training
    %       train_objects_pos      : index for train_objects
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
    %       validation_curr_paths  : Cell{validation_queue_size} contains paths
    %                                currently loaded on each batch of the queue
    %
    %% SETTERS
    %       set_async_queue_size   : Sets the size of train and validation
    %                                queues and starts matlab's pool
    %       set_train_objects      : Gets 
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   20/5/2015
    %       FOR:    VISION TEAM - AUTH
    properties  
        net_structure;

        normalization_type = 'zero_one_scale';
        mean             = [];
        batched_mean     = [];
        std              = [];
        batched_std      = [];
        
        crop_freq        = 0;
        skew_freq        = 0;
        rot_freq         = 0;
        projections_freq = 0;
        use_flipped      = 1;
        
        rot_theta_bounds = [0 0];
        skew_bounds      = [0 0];        
                
        train_objects          = [];
        train_objects_pool     = [];
        train_objects_pos      = 0;
        validation_objects     = [];
        validation_objects_pos = 0;

        %Training batch queue
        train_queue;
        validation_queue;        
        
        train_queue_size = 1;
        validation_queue_size = 1;
        
        train_queue_idx  = 1;
        validation_queue_idx =1;
        
        train_curr_paths;
        validation_curr_paths;
    end
    
    methods
        %% INIT
        function obj = BATCH_FACTORY(net_structure)
            obj.net_structure = net_structure;
        end

        %% SET QUEUE SIZE
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
                
        %% SET training objects paths
        function set_train_objects(obj,arg_objects)
            for i=length(arg_objects):-1:1
                obj.train_objects{i,1}      = arg_objects(i).paths;
            end
            obj.train_objects_pool = obj.train_objects;
        end
        
        %% SET validation objects paths
        function set_validation_objects(obj,arg_objects)
            obj.validation_objects       = vectorize_objects_fpaths(arg_objects);
        end
        %% Internal SET mean and std on batch during training
        function set_use_mean_std(obj,arg)
            obj.mean         = arg.mean;
            obj.std          = arg.std;
        end
        
        %% SET Whether to use mean/std for batch normalization
        function normalize_input(obj,arg_type)
            %This is usable only when metadata contain the extracted mean and std from the dataset
            arg_type=lower(arg_type);
            switch arg_type
                case 'zero_one_scale'
                    obj.normalization_type = arg_type;
                otherwise
                    obj.normalization_type = arg_type;
                    if(numel(obj.mean)~=0 && numel(obj.std)~=0)
                        obj.mean       = single(imresize(obj.mean,[obj.net_structure.objects_size(2) obj.net_structure.objects_size(1)],'bilinear','antialiasing',false));
                        obj.std        = single(imresize(obj.std ,[obj.net_structure.objects_size(2) obj.net_structure.objects_size(1)],'bilinear','antialiasing',false));
                    else
                        APP_LOG('warning','Metadata have not computed mean and std from current dataset. Falling back to zero_one_scale');
                        obj.normalization_type = 'zero_one_scale';
                        APP_LOG('warning','Normalization type set to 0-1 space');
                    end
            end
        end
        
        %% SET training objects flipping attribute
        function use_flipped_samples(obj,use_flipped)
            obj.use_flipped = use_flipped;
        end
        
        %% SET training objects random cropping attributes
        function crop(obj,freq)
            obj.crop_freq = freq;
        end

        %% SET training objects random skewing attributes
        function skew(obj,bounds,freq)
            obj.skew_bounds = bounds;            
            obj.skew_freq   = freq;
        end        
        
        %% SET training objects random rotation attributes
        function rotate(obj,theta_bounds,freq)
             obj.rot_theta_bounds = theta_bounds;
             obj.rot_freq = freq;
        end
        
        %% SET Use projections of objects
        function projections(obj,freq)
            obj.projections_freq = freq;
            if freq~=0
                APP_LOG('last_error','Projections are work in progress');
            end
        end
        
        %% Paths assignment to workers
        function assign_training_paths(obj,which_queue_id)
            for i=obj.net_structure.train_batch_size:-1:1
                class_idx = ceil(rand(1,1)*length(obj.train_objects_pool));
                local_array(i,1)=obj.train_objects_pool{class_idx}(1);
                
                obj.train_objects_pool{class_idx}(1)=[]; %remove used sample
                if isempty(obj.train_objects_pool{class_idx})
                    num = length(obj.train_objects{class_idx});
                    rand_idxs  = randperm(num,num);
                    obj.train_objects_pool{class_idx}=obj.train_objects{class_idx}(rand_idxs);
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
            obj.assign_training_paths(obj.train_queue_idx);%assign new paths            
            
            out=fetchOutputs(obj.train_queue{obj.train_queue_idx});%fetch old work

            out{1,1}=obj.input_normalization_function(out{1,1});
            out{2,1}=obj.net_structure.layers{end}.transform_labels(out{2,1});
            
            obj.async_load_current_train_batch(obj.train_queue_idx); %assign new work
            
            obj.train_queue_idx=obj.train_queue_idx+1;
            if(obj.train_queue_idx>obj.train_queue_size)
                obj.train_queue_idx=1;
            end
        end
        
        function out = create_validation_batch(obj)    
            if(~isempty(obj.validation_queue{obj.validation_queue_idx}))
                out=fetchOutputs(obj.validation_queue{obj.validation_queue_idx});
                out{1,1}=obj.input_normalization_function(out{1,1});
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
            out{1,1} = obj.input_normalization_function(out{1,1});
%            out{1,1} = obj.input_normalization_function(data);
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
            x.use_flipped       = obj.use_flipped;
            x.crop_freq         = obj.crop_freq;
            x.skew_freq         = obj.skew_freq;
            x.rot_freq          = obj.rot_freq;
            x.projections_freq  = obj.projections_freq;
            
            x.rot_theta_bounds  = obj.rot_theta_bounds;
            x.skew_bounds       = obj.skew_bounds;
            obj.train_queue{i}  = parfeval(@async_read_from_disk_and_preprocess,1,x);
%           out = async_read_from_disk_and_preprocess(x);
        end

        function out = prepare_validation_batch(obj)            
            obj.validation_objects_pos = 0;
            %Initial assumption - we are assuming that validation_batch_size is less than total validation objects.
            for i=1:obj.validation_queue_size
                obj.assign_validation_work(i);
            end
            out = obj.create_validation_batch();
        end
        
        function data_out = input_normalization_function(obj,data_in)
            batch_size = size(data_in,4);
            switch obj.normalization_type
                case 'subtract_means'
                    if size(obj.batched_mean,4)~=batch_size
                        obj.batched_mean=repmat(obj.mean,[1 1 1 batch_size]);
                    end
                    data_in =single(data_in);
                    data_out=(data_in - obj.batched_mean);
                case 'subtract_means_normalize_variances'
                    if size(obj.batched_mean,4)~=batch_size
                        obj.batched_mean=repmat(obj.mean,[1 1 1 batch_size]);
                    end
                    if size(obj.batched_std,4)~=batch_size
                        obj.batched_std =repmat(obj.std,[1 1 1 batch_size]);
                    end
                    data_in =single(data_in);
                    data_out=(data_in - obj.batched_mean)./obj.batched_std;
                case 'zero_one_scale'
                    data_out=im2single(data_in);
                otherwise
                    APP_LOG('last_error','Erroneous type of normalization');
            end
        end
    end    
end

