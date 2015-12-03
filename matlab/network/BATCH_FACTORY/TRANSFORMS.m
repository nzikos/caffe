classdef TRANSFORMS < handle
    %TRANSFORMS Class is used to perform transformations on training set
    %samples.
    %
    %   This class is part of BATCH_FACTORY() class.    
    %    
    %% Processing details
    % Since CNNs are invariant in many transformations, I.e. if a neural net
    % sees for the first time a human face rotated by 30 degrees it may not
    % recognize it, certain affine transformations are applied creating much
    % more unique images for the neural net, to learn.
    %
    %   Supported transformations are:
    %   1. flip
    %   2. Crop
    %   3. Rotate
    %   4. Skew
    %   5. Projective
    %   6. Jitter (todo)
    %
    %   crop_freq              : Ratio of objects random cropping per batch.
    %   rot_freq               : Ratio of objects rotation per batch.
    %   skew_freq              : Ratio of skewed objects per batch.
    %   projections_freq       : Ratio of projected objects per batch.
    %      
    %   rot_theta_bounds       : [minimum_theta maximum_theta] in degrees.
    %   skew_bounds            : [minimum_theta maximum_theta] in degrees.
    %
    %   use_flipped            : use horizontally flipped objects or not (0/1).    
    %% SETTERS
    %
    %   set_tform              : Sets the parameters of a specific
    %                            transformation.
    %% METHODS
    %
    %   apply_tforms           : Applies random transformations in a serial
    %                            manner. Samples should be inserted 
    %                            randomly in order to avoid the appliance
    %                            of the same transformation to the same
    %                            class.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   17/11/2015
    %  FOR:    VISION TEAM - AUTH    
    properties
        crop_freq        = 0;
        skew_freq        = 0;
        rot_freq         = 0;
        projs_freq       = 0;
        use_flipped      = 0;
        
        rot_theta_bounds = [0 0];
        skew_bounds      = [0 0];  
        
        
    end
    
    methods
        %% Constructor
        function obj = TRANSFORMS()
        end
        
        %% SETTERS
        function set_tform(obj,type,args)
            type = lower(type);
            switch type
                case 'flip'
                    if ~(length(args)==1 && (args{1}==1 || args{1}==0))
                        APP_LOG('last_error','Horizontal flip transformation requires 1 boolean argument');
                    end
                    obj.use_flipped=args{1};
                case 'rotate'
                    if ~(length(args)==2)
                        APP_LOG('last_error','Rotation transformations require 2 arguments. [bounds],frequency');
                    end
                    obj.rot_theta_bounds=args{1};
                    obj.rot_freq=args{2};
                case 'skew'
                    if ~(length(args)==2)
                        APP_LOG('last_error','Skew transformations require 2 arguments. [bounds],frequency');
                    end
                    obj.skew_bounds=args{1};
                    obj.skew_freq=args{2};
                case 'crop'
                    if ~(length(args)==1 && args{1}<=1 && args{1}>=0)
                        APP_LOG('last_error','Crops transformation requires only frequency per batch');
                    end
                    obj.crop_freq=args{1};
                case 'projections'
                    if ~(length(args)==1 && args{1}<=1 && args{1}>=0)
                        APP_LOG('last_error','Projective transformation requires only frequency per batch');
                    end
                    obj.projs_freq=args{1};  
                otherwise
                    APP_LOG('last_error','Unsupported Transformations, consider reading documentation');
            end
        end
        
        %% METHODS
        function processed_data = apply_tforms(obj,data,input_dims)
            
            batch_size        = size(data,4);
            % Convert frequencies to batch regions
            inner_counter     = 0;
            
            crops_reg         = [inner_counter+1 ; inner_counter+round(batch_size*obj.crop_freq)];
            inner_counter     = inner_counter +round(batch_size*obj.crop_freq);
            
            rots_reg          = [inner_counter+1 ; inner_counter+round(batch_size*obj.rot_freq)];
            inner_counter     = inner_counter +round(batch_size*obj.rot_freq);
            
            skew_reg          = [inner_counter+1 ; inner_counter+round(batch_size*obj.skew_freq)];
            inner_counter     = inner_counter +round(batch_size*obj.skew_freq);
            
            projs_reg         = [inner_counter+1 ; inner_counter+round(batch_size*obj.projs_freq)];
            inner_counter     = inner_counter +round(batch_size*obj.projs_freq);
            
            non_processed_reg = [inner_counter+1 ; batch_size];         
            % PERFORM PROCESSING
            if crops_reg(1)<=crops_reg(2)
                object_dims = size(data);
                if object_dims(1)~=input_dims(1) || object_dims(2)~=input_dims(2)
                    processed_data(:,:,:,crops_reg(1):crops_reg(2)) = get_random_crops(data(:,:,:,crops_reg(1):crops_reg(2)),input_dims);
                else
                    APP_LOG('error',0,'Identical dimensions detected while extracting random segments');
                    APP_LOG('error',0,'Object dimensions are [%d,%d]',object_dims(1),object_dims(2));
                    APP_LOG('last_error',0,'Network dimensions are [%d,%d]',input_dims(1),input_dims(2));
                end
            end
            if rots_reg(1)<=rots_reg(2)
                processed_data(:,:,:,rots_reg(1):rots_reg(2)) = get_random_rotations(obj.rot_theta_bounds,data(:,:,:,rots_reg(1):rots_reg(2)),input_dims);
            end
            if skew_reg(1)<=skew_reg(2)
                processed_data(:,:,:,skew_reg(1):skew_reg(2)) = get_random_skews(obj.skew_bounds,data(:,:,:,skew_reg(1):skew_reg(2)),input_dims);
            end
            if projs_reg(1)<=projs_reg(2)
                processed_data(:,:,:,projs_reg(1):projs_reg(2))=get_projections(data(:,:,:,projs_reg(1):projs_reg(2)),input_dims);
            end
            if non_processed_reg(1)<=non_processed_reg(2)
                processed_data(:,:,:,non_processed_reg(1):non_processed_reg(2)) = get_resized_objects(data(:,:,:,non_processed_reg(1):non_processed_reg(2)),input_dims);
            end            
            % PERFORM HORIZONTAL FLIPPING
            if(obj.use_flipped)
                rnd_flip_idxs=randperm(batch_size,floor(0.5*batch_size));
                processed_data(:,:,:,rnd_flip_idxs)=flip(processed_data(:,:,:,rnd_flip_idxs),1); %Width is 1st dimension, so we are actually performing vertical flip on a 90degrees rotated object
            end
        end
    end
    
end

