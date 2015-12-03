classdef INPUT_NORM < handle
    %INPUT_NORM  Class is used to perform data normalization on samples.
    %
    %   This class is part of BATCH_FACTORY() class.    
    %
    %%  PROPERTIES
    %
    %       normalization_type: Used to declare the type of normalization 
    %                           which will be applied before the samples 
    %                           will be inserted to Network.
    %                           data normalization types are:
    %                           1. Zero_one_scale -> [0 1]
    %                           2. Subtract means -> X-E(D)
    %                           3. Subtract_mean_normalize_variances -> [X-E(D)]/std(D)
    %       
    %       mean              : Used to store the pixelwise mean as
    %                           extracted from extraction_model.
    %
    %       std               : Used to store the pixelwise std as
    %                           extracted from extraction_model.
    %
    %       batched_mean      : internal var to perform faster
    %                           subtractions.
    %
    %       batched_std       : internal var to perform faster divisions.
    %
    %       epsilon           : used to avoid division by zero in case of
    %                           uniformly distributed data accross a
    %                           specific pixel through the training dataset.
    %
    %%  SETTERS
    %
    %       set_mean_std          : Sets the mean,std properties.
    %
    %       set_epsilon           : Sets epsilon properties.
    %
    %       set_norm_type         : Sets normalization type.
    %
    %% METHODS
    %
    %       do_normalization      : Performs normalization on batched samples.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   17/11/2015
    %  FOR:    VISION TEAM - AUTH    
    properties
        normalization_type = 'none';
        mean             = [];
        batched_mean     = [];
        std              = [];
        batched_std      = [];
        epsilon          = 1e-8;
    end
    
    methods
        %% CONSTRUCTOR
        function obj = INPUT_NORM()
        end
        
        %% SETTERS
        %Internal SET mean and std on batch during training
        function set_mean_std(obj,arg)
            obj.mean         = arg.mean;
            obj.std          = arg.std;
        end
        function set_epsilon(obj,eps)
            obj.epsilon = eps;
        end
        %SET Whether to use mean/std for batch normalization
        function set_norm_type(obj,arg_type,input_dims)
            %This is usable only when metadata contain the extracted mean and std from the dataset
            arg_type=lower(arg_type);
            switch arg_type
                case 'zero_one_scale'
                    obj.normalization_type = arg_type;
                case 'none'
                    obj.normalization_type = arg_type;
                case 'subtract_means'
                    obj.normalization_type = arg_type;
                    if(numel(obj.mean)~=0 && numel(obj.std)~=0)
                        obj.mean       = single(imresize(obj.mean,[input_dims(2) input_dims(1)],'bilinear','antialiasing',false));
                        obj.std        = single(imresize(obj.std ,[input_dims(2) input_dims(1)],'bilinear','antialiasing',false));
                    else
                        APP_LOG('last_error','Metadata have not computed mean and std from current dataset.');
                    end                    
                case 'subtract_means_normalize_variances'
                    obj.normalization_type = arg_type;
                    if(numel(obj.mean)~=0 && numel(obj.std)~=0)
                        obj.mean       = single(imresize(obj.mean,[input_dims(2) input_dims(1)],'bilinear','antialiasing',false));
                        obj.std        = single(imresize(obj.std ,[input_dims(2) input_dims(1)],'bilinear','antialiasing',false));
                    else
                        APP_LOG('last_error','Metadata have not computed mean and std from current dataset.');
                    end                    
                otherwise
                    APP_LOG('last_error','Unknown samples normalization type.');
            end
        end
        %% METHODS
        function data_out = do_normalization(obj,data_in)
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
                    data_out=(data_in - obj.batched_mean)./(obj.batched_std + obj.epsilon);
                case 'zero_one_scale'
                    data_out=im2single(data_in);
                case 'none'
                    data_out=single(data_in);
                otherwise
                    APP_LOG('last_error','Unknown samples normalization type.');
            end
        end        
    end
    
end

