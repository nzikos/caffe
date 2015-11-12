classdef NET_STRUCTURE < handle
    %NET_STRUCTURE Defines an object that holds the network structure.
    %The object is responsible to create the .prototxt to feed into Caffe
    %in order to create the network
    %Layers supported are:
    %   1. Convolution
    %   2. Pooling
    %   3. Activation
    %   4. InnerProduct
    %   5. MVN
    %   6. LRN
    %   7. Dropout
    %
    %WEIGHT/BIAS FILLERS ARE:
    %   1. constant
    %   2. gaussian
    %   3. uniform
    %BIAS_TERM determines whether a bias or not to be used
    %
    %ACTIVATION FUNCTIONS ARE: 
    %   1. ReLU
    %   2. LReLU
    %   3. PReLU
    %   4. Sigmoid
    %   5. Tanh
    %ERROR FUNCTIONS ARE:
    %   1. SoftmaxWithLoss
    %   2. SigmoidCrossEntropyLoss
    %   3. EuclideanLoss
    %   4. HingeLoss
    
    %BOTTOM-TOP BLOBS are automatically defined, in-place transformations
    %are also automatically defined.
    %
    %% AUTHOR: PROVOS ALEXIS
    %   DATE:   08/11/2015
    %   FOR:    VISION TEAM - AUTH
    properties
        prototxt_path       = [];
        train_batch_size    = -1;
        val_batch_size      = -1;
        test_batch_size     = -1;
        
        objects_size        = [-1,-1,-1];
      
        valid_structure     = -1;
        
        layers              = {};
        params              = struct('data',[],'name',[]);
        sub_counter         = 1;
        
        m_mult;
        lr_mult;
        wd_mult;
    end
    
    methods
        function structure = NET_STRUCTURE(path)
            handle_dir(path,'throw error');
            structure.prototxt_path = path;
        end
        %% ADD LAYER
        function add_layer(structure,type,varargin)
            
            if isempty(structure.params(1).name)
	            params_counter = 1;
            else
            	params_counter=length(structure.params)+1;
            end
	            
            switch type
                case 'Convolution'
                    structure.sub_counter   = 1;
                    name = ['conv_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if length(varargin)<2
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end
                case 'InnerProduct'
                    structure.sub_counter   = 1;
                    name = ['ip_' num2str(params_counter) '_' num2str(structure.sub_counter)];                    
                    if length(varargin)<2
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end
                case 'PReLU'
                    structure.sub_counter   = 1;
                    name = ['prelu_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if length(varargin)<2
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end
                case 'Pooling'
                    structure.sub_counter   = structure.sub_counter + 1;
                    name = ['pool_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if isempty(varargin)
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end
                case 'MVN'
                    structure.sub_counter   = structure.sub_counter + 1;
                    name = ['MVN_' num2str(params_counter) '_' num2str(structure.sub_counter)];                    
                    if isempty(varargin)
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end                    
                case 'LRN'
                    structure.sub_counter   = structure.sub_counter + 1;
                    name = ['LRN_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if isempty(varargin)
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end                    
                case 'Activation'
                    structure.sub_counter   = structure.sub_counter + 1;
                    switch( varargin{1}{1} )
                        case 'ReLU'
                            name = ['relu_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                        case 'Sigmoid'
                            name = ['sigmoid_' num2str(params_counter) '_' num2str(structure.sub_counter)];                    
                        case 'Tanh'
                            name = ['tanh_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                        otherwise
                            APP_LOG('last_error','Unknown activation type %s. Use ReLU,Sigmoid,Tanh');
                    end
                case 'Dropout'
                    structure.sub_counter   = structure.sub_counter + 1;
                    name = ['drop_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if isempty(varargin)
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end                    
                case 'Output'
                    structure.sub_counter   = structure.sub_counter + 1;                    
                    name = ['output_' num2str(params_counter) '_' num2str(structure.sub_counter)];
                    if isempty(varargin)
                        APP_LOG('last_error','%s: Erroneous initialization. Consider reading the documentation',name);
                    end                    
                otherwise
                    APP_LOG('last_error','Unknown layer type %s. Use ReLU,Sigmoid,Tanh',type);                    
            end
            
            [bottom,bottom_size,top] = structure.get_bottom_top_blobs(name);
            local_m_mult             = [];            
            local_lr_mult            = [];
            local_wd_mult            = [];
            local_params             = [];
            switch type
                case 'Convolution'
                    %INITIALIZE LAYER'S HYPER-PARAMETERS
                    layer = Convolution_layer(bottom,bottom_size,top,name,varargin{1});
                                        
                    %INITIALIZE LAYER'S LOCAL MULTIPLIERS
                    [local_m_mult,local_lr_mult,local_wd_mult] = layer.process_local_multipliers(varargin{2});
                    
                    %FILL PARAMS
                    if length(varargin)==3
                        local_params.data = layer.fill_params(varargin{3});
                    else
                        APP_LOG('warning','It is strongly recommended to initialize %s layer''s parameters',layer.name);
                        local_params.data = layer.fill_params({{'constant',0},{'constant',0}});
                    end
                    local_params.name = name;
                    
                case 'InnerProduct'
                    %INITIALIZE LAYER'S HYPER-PARAMETERS
                    layer = InnerProduct_layer(bottom,bottom_size,top,name,varargin{1});
                    
                    %INITIALIZE LAYER'S LOCAL MULTIPLIERS
                    [local_m_mult,local_lr_mult,local_wd_mult] = layer.process_local_multipliers(varargin{2});
                    
                    %FILL PARAMS                    
                    if length(varargin)==3
                        local_params.data = layer.fill_params(varargin{3});
                    else
                        APP_LOG('warning','It is strongly recommended to initialize %s layer''s parameters',layer.name);
                        local_params.data = layer.fill_params({{'constant',0},{'constant',0}});
                    end
                    local_params.name = name;
                    
                case 'PReLU'
                    %INITIALIZE LAYER'S HYPER-PARAMETERS                    
                    layer = PReLU_layer(bottom,bottom_size,top,name,varargin{1});

                    %INITIALIZE LAYER'S LOCAL MULTIPLIERS
                    [local_m_mult,local_lr_mult,local_wd_mult] = layer.process_local_multipliers(varargin{2});
                    
                    %FILL PARAMS                    
                    if length(varargin)==3
                        local_params.data = layer.fill_params(varargin{3});
                    else
                        APP_LOG('debug','%s layer''s parameters under default initialization',layer.name);
                        local_params.data = layer.fill_params({'constant',0.25});
                    end
                    local_params.name = name;
                case 'Pooling'
                    layer = pooling_layer(bottom,bottom_size,top,name,varargin{1});
                case 'MVN'
                    layer = MVN_layer(bottom,bottom_size,top,name,varargin{1});
                case 'LRN'
                    layer = LRN_layer(bottom,bottom_size,top,name,varargin{1});
                case 'Activation'
                    layer = Activation_layer(bottom,bottom_size,bottom,name,varargin{1});
                case 'Dropout'
                    layer = Dropout_layer(bottom,bottom_size,bottom,name,varargin{1});
                case 'Output'
                    layer = OUTPUT_LAYER(bottom,bottom_size,top,name,varargin{1});
                otherwise
                    APP_LOG('last_error','Unknown layer type: %s',type);
            end
            
            %APPEND layer
            if isempty(structure.layers)
                structure.layers{1,1}=layer;
            else
                structure.layers{end+1,1}=layer;
            end
            %APPEND local_multiplies
            if ~(isempty(local_lr_mult))
                if isempty(structure.lr_mult)
                    structure.m_mult(1,:)       = local_m_mult(1,:);
                    structure.lr_mult(1,:)      = local_lr_mult(1,:);
                    structure.wd_mult(1,:)      = local_wd_mult(1,:);                    
                else
                    structure.m_mult(end+1,:)   = local_m_mult(1,:);
                    structure.lr_mult(end+1,:)  = local_lr_mult(1,:);
                    structure.wd_mult(end+1,:)  = local_wd_mult(1,:);                    
                end
                if length(structure.lr_mult)~=length(structure.m_mult) || length(structure.lr_mult)~=length(structure.wd_mult)
                    APP_LOG('last_error','Local multipliers length missmatch');
                end
            end
            %APPEND params
            if ~(isempty(local_params))
                if isempty(structure.params(1).name)
                    structure.params(1)     = local_params;
                else
                    structure.params(end+1) = local_params;
                end
            end
            structure.valid_structure = -1;            
        end
        
        %% SET BATCH SIZES
        function set_batch_size(structure,phase,batch_size)
            switch phase
                case 'train'
                    structure.train_batch_size = batch_size;
                case 'validation'
                    structure.val_batch_size   = batch_size;
                case 'test'
                    structure.test_batch_size  = batch_size;
                otherwise
                    APP_LOG('last_error','PHASE: %s does not exist, use train/validation/test');
            end
            structure.valid_structure = -1;
        end
        %% SET OBJECTS DIMENSIONS
        function set_objects_dims(structure,height,width,depth)
            structure.objects_size    = [height, width, depth];
            structure.valid_structure = -1;
        end
        
        %% DEFINE BOTTOM/TOP BLOBS OF SPECIFIC LAYER
        function [bottom,bottom_size,top] = get_bottom_top_blobs(structure,name)
            if isempty(structure.layers)
                bottom       = 'data';
                bottom_size  = structure.objects_size;
            else
                bottom       = structure.layers{end}.top;
                bottom_size  = structure.layers{end}.top_size;
            end
            top = name;
            tmp = {'Height','Width','Depth'};
            if ~all(bottom_size>0)
                APP_LOG('last_error','%s: bottom %s not set or LE than zero',name,tmp{find(structure.objects_size>0)});
            end            
        end
        %% PRINT TO FILE
        function filepath = create_prototxt(structure,phase)
            name = sprintf('%s_structure.prototxt',phase);
            filepath=fullfile(structure.prototxt_path,name);
            APP_LOG('info','Creating prototxt %s',filepath);
            if structure.valid_structure~=1
                structure.validate_structure();
            end
            fileID = fopen(filepath,'w');
            fprintf(fileID,'name: "%s"\n',phase);
            fprintf(fileID,'input: "data"\n');
            switch(phase)
                case 'train'
                    fprintf(fileID,'input_dim: %d\n',structure.train_batch_size);
                case 'validation'
                    fprintf(fileID,'input_dim: %d\n',structure.val_batch_size);
                case 'test'
                    fprintf(fileID,'input_dim: %d\n',structure.test_batch_size);
                otherwise
                    APP_LOG('last_error','Phase "%s" does not exist, use train/validation/test');
            end
            fprintf(fileID,'input_dim: %d\n',structure.objects_size(3));
            fprintf(fileID,'input_dim: %d\n',structure.objects_size(1));
            fprintf(fileID,'input_dim: %d\n',structure.objects_size(2));
            if(strcmp(phase,'train')==1)
                labels_size = structure.layers{end}.labels_size;
                fprintf(fileID,'input: "label"\n');
                fprintf(fileID,'input_dim: %d\n',structure.train_batch_size);
                fprintf(fileID,'input_dim: %d\n',labels_size(3));
                fprintf(fileID,'input_dim: %d\n',labels_size(1));
                fprintf(fileID,'input_dim: %d\n',labels_size(2));
            end
            for i=1:length(structure.layers)
                layer=structure.layers{i};
                layer.print_layer(fileID,phase);
            end
            fclose(fileID);
        end
        %% PRINT FEATURE MAPS
        function validate_structure(structure)
            if structure.train_batch_size <= 0
                APP_LOG('last_error','Training batch size not set');
            end
            if structure.val_batch_size <= 0
                APP_LOG('last_error','Validation batch size not set');
            end
            if structure.test_batch_size <= 0
                APP_LOG('last_error','Test batch size not set');
            end
            
            tmp = {'Height','Width','Depth'};
            if ~all(structure.objects_size>0)
                APP_LOG('last_error','Objects %s not set',tmp{find(structure.objects_size>0)});
            end
            if ~all(structure.layers{end}.labels_size>0)
                APP_LOG('last_error','Labels %s not set',tmp{find(structure.layers{end}.labels_size>0)});
            end
            
            APP_LOG('info','Outputs And Weights/Bias');
            APP_LOG('info','###. %15s %6s%7s%6s%10s%5s','LAYER NAME','HEIGHT','HEIGHT','DEPTH','WEIGHTS','BIAS');
            APP_LOG('info','%3d. %15s:[%5d,%5d,%5d]',0,'DATA',structure.objects_size(1),structure.objects_size(2),structure.objects_size(3));

            for i=1:length(structure.layers)
                layer=structure.layers{i};
                weights =0;
                bias    =0;
                switch class(layer)
                    case 'Convolution_layer'
                        weights = layer.weights_size(1)*layer.weights_size(2)*layer.weights_size(3)*layer.weights_size(4);
                        if layer.bias_term
                            bias  =  layer.bias_size(1)*layer.bias_size(2)*layer.bias_size(3)*layer.bias_size(4);
                        end
                    case 'InnerProduct_layer'
                        weights = layer.weights_size(1)*layer.weights_size(2)*layer.weights_size(3)*layer.weights_size(4);
                        if layer.bias_term
                            bias  =  layer.bias_size(1)*layer.bias_size(2)*layer.bias_size(3)*layer.bias_size(4);
                        end
                    case 'PReLU_layer'
                        weights   = 1 + (1-layer.channel_shared)*(layer.bottom_size(3)-1);
                end
                if weights~=0 || bias~=0
                    APP_LOG('info','%3d. %15s:[%5d,%5d,%5d]%10d%5d',i,layer.name,layer.top_size(1),layer.top_size(2),layer.top_size(3),weights,bias);
                else
                    APP_LOG('info','%3d. %15s:[%5d,%5d,%5d]',i,layer.name,layer.top_size(1),layer.top_size(2),layer.top_size(3));
                end
            end           
            APP_LOG('debug','Structure validity test passed');
            structure.valid_structure=1;
        end
end

end
