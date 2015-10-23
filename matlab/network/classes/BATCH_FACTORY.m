classdef BATCH_FACTORY < handle
    %BATCH_FACTORY class purpose is to create a batch of images in order to
    %be processed by caffe.
    %% ATTRIBUTES
    %       train_batch_size       : The number of training objects which
    %                                are,at once, send to caffe while training.
    %       val_batch_size         : The number of validation objects which are, 
    %                                at once, send to caffe while validating.
    %       test_batch_size        : The number of testing images which are, at
    %                                once, send to caffe while testing.
    %       input_dims             : The size [HxW] of the images caffe expects 
    %                                to get.
    %       use_mean_std           : Use mean and std extracted from
    %                                training set in order to normalize
    %                                samples. mean and std dims are [HxWxD]
    %       use_flip               : Use flipped objects or not.
    %       use_random_segments    : Use random segments from the objects or
    %                                not.
    %       use_rot                : Use rotated images or not.
    %       use_projs              : Use projections (projective2d function)
    %
    %       freq_random_segments   : Ratio of objects random segmentation per
    %                                batch
    %       rot_freq               : Ratio of objects rotation per batch
    %       projections_freq       : Frequency of projected objects per batch
    %       
    %       rot_angle_bounds       : [minimum_angle maximum_angle] in degrees
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

        use_mean_std     = 0;
        mean             = [];
        std              = [];
        
        use_flip         = 0;
        use_random_segments = 0;        
        use_rot          = 0;
        use_projs        = 0;
        
        freq_random_segments= 0;        
        rot_freq         = 0;
        projections_freq = 0;
        
        rot_angle_bounds = [0 0];
                
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
%            rand_idxs         = randperm(length(obj.train_objects),length(obj.train_objects));
%            obj.train_objects = obj.train_objects(rand_idxs);
        end
        
        %% SET validation objects paths
        function set_validation_objects(obj,arg_objects)
            obj.validation_objects       = vectorize_objects_fpaths(arg_objects);
        end
        %% Internal SET use mean std on batch during training
        function set_use_mean_std(obj,arg)
            obj.use_mean_std = arg.get_mean_std;
            obj.mean         = arg.mean;
            obj.std          = arg.std;
        end
        
        %% SET Whether to use mean/std for batch normalization
        function normalize_batches(obj,arg)
            %This is usable only when metadata contain the extracted mean and std from the dataset
            if obj.use_mean_std && ~arg    
                obj.use_mean_std = arg;
                if obj.use_mean_std
                    obj.mean       = single(imresize(obj.mean,[obj.net_structure.object_height obj.net_structure.object_width] ,'bilinear','antialiasing',false));
                    obj.std        = single(imresize(obj.std ,[obj.net_structure.object_height obj.net_structure.object_width],'bilinear','antialiasing',false));
                end            
            else
                APP_LOG('warning','Metadata have not computed mean and std from current dataset. Ignoring Force use');
            end
        end
        
        %% SET training objects flipping attribute
        function use_flipped(obj,use_flip)
            obj.use_flip = use_flip;
        end
        
        %% SET training objects random segmentation attributes
        function use_random_segmentation(obj,value,freq)
            obj.use_random_segments=value;
            if (obj.use_random_segments)
                obj.freq_random_segments = freq;
            end
        end
        
        %% SET training objects rotation attributes
        function use_rotation(obj,value,theta_bounds,freq)
            obj.use_rot = value;
            if (obj.use_rot)
                obj.rot_angle_bounds = theta_bounds;
                obj.rot_freq = freq;
            end
        end
        
        %% SET Use projections of objects
        function use_projections(obj,value,freq)
            obj.use_projs = value;
            obj.projections_freq = freq;
        end
        
        %% Paths assignment to workers
        function assign_training_paths(obj,which_queue_id)
            %from = obj.train_objects_pos;
            %to   = obj.train_objects_pos + obj.net_structure.train_batch_size;
            %obj.train_objects_pos = to;
            %obj.train_curr_paths{which_queue_id} = obj.train_objects(from+1:to);
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
            if(obj.use_mean_std)
                out{1,1}=single(out{1,1});
                for i=1:size(out{1,1},4)
                    out{1,1}(:,:,:,i)=(out{1,1}(:,:,:,i)-obj.mean)./obj.std;
                end
            else
                out{1,1}=im2single(out{1,1});
            end
            out{2,1}=single(out{2,1});
            
            obj.async_load_current_train_batch(obj.train_queue_idx); %assign new work
            
            obj.train_queue_idx=obj.train_queue_idx+1;
            if(obj.train_queue_idx>obj.train_queue_size)
                obj.train_queue_idx=1;
            end
        end
        
        function out = create_validation_batch(obj)    
            if(~isempty(obj.validation_queue{obj.validation_queue_idx}))
                out=fetchOutputs(obj.validation_queue{obj.validation_queue_idx});
                if(obj.use_mean_std)
                    out{1,1}=single(out{1,1});                    
                    for i=1:size(out{1,1},4)
                        out{1,1}(:,:,:,i)=(out{1,1}(:,:,:,i)-obj.mean)./obj.std;
                    end
                else
                    out{1,1}=im2single(out{1,1});
                end
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
        
        %% Prepare asynchronous Batch creation
        function prepare_train_batch(obj) 
            %We are assuming that batch size is smaller than number of objects 
            for i=1:obj.train_queue_size;
                obj.assign_training_paths(i);
                obj.async_load_current_train_batch(i);
            end
        end
        
        function async_load_current_train_batch(obj,i)
            x.paths                    = obj.train_curr_paths{i};
            x.input_dims               = [obj.net_structure.object_height obj.net_structure.object_width];
            x.use_flip                 = obj.use_flip;
            x.use_rot                  = obj.use_rot;
            x.use_random_segments      = obj.use_random_segments;
            x.use_projs                = obj.use_projs;
            x.freq_random_segments     = obj.freq_random_segments;
            x.rot_freq                 = obj.rot_freq;
            x.projections_freq         = obj.projections_freq;
            
            x.rot_angle_bounds         = obj.rot_angle_bounds;
            obj.train_queue{i}         = parfeval(@async_read_from_disk_and_preprocess,1,x);
        end

        function out = prepare_validation_batch(obj)            
            obj.validation_objects_pos = 0;
            %Initial assumption - we are assuming that validation_batch_size is less than total validation objects.
            for i=1:obj.validation_queue_size
                obj.assign_validation_work(i);
            end
            out = obj.create_validation_batch();
        end
    end
end

