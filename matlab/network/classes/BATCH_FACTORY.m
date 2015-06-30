classdef BATCH_FACTORY < handle
    %BATCH_FACTORY class purpose is to create a batch of images in order to
    %be processed by caffe.
    %       stride     : the offset between 2 random segments in
    %                    height and width.
    %       batch_size : The number of images which are send to caffe.
    %       
    %       input_dims : The size [HxW] of the images caffe expects to get.
    %       flip       : use flipped images or not
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   20/5/2015
    %       FOR:    VISION TEAM - AUTH
    properties  
        create_training_batch   =[];
        create_validation_batch =[];

        stride       = [];
        batch_size   = [];
        input_dims   = [];
        flip         = 0;        
    end
    
    methods
        %% INIT
        function batch_factory = BATCH_FACTORY()
            batch_factory.create_training_batch  =@(paths)create_random_batch(paths,batch_factory.batch_size,batch_factory.flip,batch_factory.input_dims);
            batch_factory.create_validation_batch=@(paths)create_simple_batch(paths,batch_factory.batch_size,batch_factory.input_dims);
            batch_factory.stride = 1;
        end
        
        %% SETTERS
        
        function set_batch_size(batch_factory,batch_size)
            batch_factory.batch_size = batch_size;
        end
        
        function set_input_size(batch_factory,input_size)
            batch_factory.input_dims = input_size;
        end
        
        function use_flipped(batch_factory,flip)
            batch_factory.flip = flip;
        end
        
        function set_training_batch_method(batch_factory,method,options)
            switch(method)
                case 'random'
                    batch_factory.create_training_batch  =@(paths)create_random_batch(paths,batch_factory.batch_size,batch_factory.flip,batch_factory.input_dims);
                case 'random_segments'
                    batch_factory.create_training_batch  =@(paths)create_random_batch_w_rnd_sgmnt(paths,batch_factory.batch_size,batch_factory.flip,batch_factory.input_dims,batch_factory.stride);
                    batch_factory.stride=options{1};
                otherwise
                    APP_LOG('error_last',0,'Batch creation method "%s" is not supported',method);
            end
        end
        
    end
end

