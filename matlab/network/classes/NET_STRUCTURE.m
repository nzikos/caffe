classdef NET_STRUCTURE < handle
    %NET_STRUCTURE Defines an object that holds the network structure.
    %The object is responsible to create the .prototxt to feed into Caffe
    %in order to create the network
    %WEIGHT/BIAS FILLERS ARE: CONSTANT, XAVIER, GAUSSIAN
    %BIAS_TERM determines wheter a bias or not to be used
    %ACTIVATION FUNCTIONS ARE: ReLU, SIGMOID, TANH
    %ERROR FUNCTIONS ARE: SOFTMAX_LOSS, SIGMOID_CROSS_ENTROPY_LOSS
    
    %BOTTOM-TOP BLOBS are automatically defined, in-place transformations
    %are also automatically defined.
    %% AUTHOR: PROVOS ALEXIS
    %   DATE:   07/10/2015
    %   FOR:    VISION TEAM - AUTH
    properties       
        prototxt_path       = [];
        train_batch_size    = -1;
        val_batch_size      = -1;
        test_batch_size     = -1;
        
        object_depth        = -1;
        object_height       = -1;
        object_width        = -1;
        labels_length       = -1;
        
        n_layers            = 0;
        counter             = 0;
        valid_structure     = -1;
        
        map = struct('num',[],'name',[],'type',[],'bottom',[],'top',[],'hyper_params',[],'output',[],'parameters',[]);
        lr_mult;
    end
    
    methods
        function structure = NET_STRUCTURE(path)
            structure.prototxt_path = path;
        end

        %% PRINT WEIGHTS/BIAS FILLER PARAMETERS
        function parameters = print_filler_parameters(structure,type,value)
            switch(type)
                case 'gaussian'
                    formatSpec =['               type: "gaussian"\n'...
                                 '               std: %1.10f\n'];
                    parameters = sprintf(formatSpec,value);
                case 'xavier'
                    formatSpec ='               type: "xavier"\n';
                    parameters = sprintf(formatSpec);
                case 'constant'
                    formatSpec =['               type: "constant"\n'...
                                 '               value: %1.10f\n'];
                    parameters = sprintf(formatSpec,value);
                otherwise
                    APP_LOG('last_error','Unknown filler type "%s"',type);
            end
        end

        function text = print_weight_bias_term_filler(structure,hyper_params)
            weights_filler = structure.print_filler_parameters(hyper_params.weight_filler_type,hyper_params.weight_filler_value);
            if hyper_params.bias_term
                bias_filler = structure.print_filler_parameters(hyper_params.bias_filler_type,hyper_params.bias_filler_value);
                formatSpec =['          bias_term: true\n'...
                             '          bias_filler {\n'...
                             '%s'...
                             '          }\n'...
                             '          weight_filler {\n'...
                             '%s'...
                             '          }\n'];
                text = sprintf(formatSpec,bias_filler,weights_filler);
            else
                formatSpec =['          bias_term: false\n'...
                             '          weight_filler {\n'...
                             '%s'...
                             '          }\n'];                    
                text = sprintf(formatSpec,weights_filler);                                     
            end
        end        
        
        %% SET BATCH SIZES
        function set_batch_size(structure,phase,batch_size)
            switch phase
                case 'train'
                    structure.train_batch_size = batch_size;
                case 'validation'
                    structure.val_batch_size = batch_size;
                case 'test'
                    structure.test_batch_size = batch_size;        
                otherwise
                    APP_LOG('last_error','PHASE: %s does not exist, use train/validation/test');
            end
            structure.valid_structure = -1;
        end
        %% SET LABELS LENGTH
        function set_labels_length(structure,labels_length)
            structure.labels_length = labels_length;
            structure.valid_structure = -1;            
        end
        %% SET OBJECT DIMENSIONS
        function set_input_object_dims(structure,depth,height,width)
            structure.object_depth  = depth;
            structure.object_height = height;
            structure.object_width  = width;
            structure.valid_structure = -1;            
        end        
        
        %% DEFINE BOTTOM/TOP BLOBS OF SPECIFIC LAYER
        function [bottom,top] = get_bottom_top_blobs(structure,name)
            current_sublayer = structure.counter + 1;
            if current_sublayer==1
                bottom='data';
            else
                bottom=structure.map(current_sublayer-1).top;
            end
            top = name;
        end
        %% ADD CONVOLUTIONAL LAYER
        function add_CONV_layer(structure,name,number_of_outputs,kernel,...
                                stride,pad,bias_term,weight_filler_type,...
                                weight_filler_value,weight_lr_mult,...
                                bias_filler_type,bias_filler_value,bias_lr_mult)

            s.num_output            = number_of_outputs;
            switch length(kernel)
                case 1
                    s.kernel_h = kernel;
                    s.kernel_w = kernel;
                case 2
                    s.kernel_h = kernel(1);
                    s.kernel_w = kernel(2);                    
                otherwise
                    APP_LOG('last_error','Erroneous kernel dimensions for %s. Use single value or array of 2 values [k_h k_w]');
            end
            switch length(pad)
                case 1
                    s.pad_h     = pad;
                    s.pad_w     = pad;
                case 2
                    s.pad_h     = pad(1);
                    s.pad_w     = pad(2);
                otherwise
                    APP_LOG('last_error','Erroneous padding dimensions for %s. Use single value or array of 2 values [p_h p_w]');
            end
            switch length(stride)
                case 1
                    s.stride_h  = stride;
                    s.stride_w  = stride;
                case 2
                    s.stride_h  = stride(1);
                    s.stride_w  = stride(2);
                otherwise
                    APP_LOG('last_error','Erroneous stride dimensions for %s. Use single value or array of 2 values [s_h s_w]');
            end
            s.bias_term           = bias_term;
            s.weight_filler_type  = weight_filler_type;
            s.weight_filler_value = weight_filler_value;
            s.weight_lr_mult      = weight_lr_mult;
            s.bias_filler_type    = bias_filler_type;
            s.bias_filler_value   = bias_filler_value;
            s.bias_lr_mult        = bias_lr_mult;
            
            [bottom,top] = structure.get_bottom_top_blobs(name);
            
            structure.put_to_map(1,name,'CONVOLUTION',bottom,top,s);
            structure.valid_structure = -1;            
        end
        %% PRINT CONVOLUTIONAL LAYER
        function print_CONV_layer(structure,fileID,layer)
            bias_weight_filler    = structure.print_weight_bias_term_filler(layer.hyper_params);
            formatSpec = ['layers {\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '      blobs_lr: %1.12f\n'...
                         '      blobs_lr: %1.12f\n'...
                         '      convolution_param {\n'...
                         '          num_output: %d\n'...
                         '          kernel_h: %d\n'...
                         '          kernel_w: %d\n'...
                         '          pad_h: %d\n'...
                         '          pad_w: %d\n'...
                         '          stride_h: %d\n'...
                         '          stride_w: %d\n'...
                         '%s'...
                         '      }\n'...
                         '}'];

            hyper_params = layer.hyper_params;

            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,layer.bottom,layer.top,...
                                    hyper_params.weight_lr_mult,hyper_params.bias_lr_mult,...
                                    hyper_params.num_output,hyper_params.kernel_h,hyper_params.kernel_w,...
                                    hyper_params.pad_h,hyper_params.pad_w,hyper_params.stride_h,...
                                    hyper_params.stride_w,bias_weight_filler);
        end
            
        %% ADD ACTIVATION LAYER        
        function add_ACTIV_layer(structure,name,type)            
            [bottom,top] = structure.get_bottom_top_blobs(name);
            type=upper(type);
            switch type
                case 'RELU'
                    structure.put_to_map(0,name,type,bottom,bottom,[]);
                case 'SIGMOID'
                    structure.put_to_map(0,name,type,bottom,top,[]);                    
                case 'TANH'
                    structure.put_to_map(0,name,type,bottom,top,[]);                    
                otherwise
                    APP_LOG('last_error','UNKNOWN Activation %s',type);
            end
            structure.valid_structure = -1;
        end

        %% PRINT ACTIVATION LAYER
        function print_ACTIV_layer(structure,fileID,layer)
            formatSpec= ['layers{\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '}'];
            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,layer.bottom,layer.top);
        end
        %% ADD LOCAL RESPONSE NORMALIZATION LAYER                
        function add_LRN_layer(structure,name,local_size,alpha,beta,norm_region)
            s.local_size            = local_size;
            s.alpha                 = alpha;
            s.beta                  = beta;
            s.norm_region           = upper(norm_region);

            [bottom,top] = structure.get_bottom_top_blobs(name);
            structure.put_to_map(0,name,'LRN',bottom,top,s);
            structure.valid_structure = -1;            
        end
        %% PRINT LOCAL RESPONSE NORMALIZATION LAYER
        function print_LRN_layer(structure,fileID,layer)
            formatSpec= ['layers{\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '      lrn_param{\n'...
                         '            local_size: %d\n'...
                         '            alpha: %1.10f\n'...
                         '            beta: %1.10f\n'...
                         '            norm_region: %s\n'...
                         '      }\n'...
                         '}'];
            hyper_params = layer.hyper_params;
            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,layer.bottom,layer.top,...
                                    hyper_params.local_size,hyper_params.alpha,...
                                    hyper_params.beta,hyper_params.norm_region);
        end
        %% ADD POOLING LAYER
        function add_POOL_layer(structure,name,method,kernel,stride,pad)
            s.method                = upper(method);
            switch length(kernel)
                case 1
                    s.kernel_h = kernel;
                    s.kernel_w = kernel;
                case 2
                    s.kernel_h = kernel(1);
                    s.kernel_w = kernel(2);                    
                otherwise
                    APP_LOG('last_error','Erroneous kernel dimensions for %s. Use single value or array of 2 values [k_h k_w]');
            end
            switch length(pad)
                case 1
                    s.pad_h     = pad;
                    s.pad_w     = pad;
                case 2
                    s.pad_h     = pad(1);
                    s.pad_w     = pad(2);
                otherwise
                    APP_LOG('last_error','Erroneous padding dimensions for %s. Use single value or array of 2 values [p_h p_w]');
            end
            switch length(stride)
                case 1
                    s.stride_h  = stride;
                    s.stride_w  = stride;
                case 2
                    s.stride_h  = stride(1);
                    s.stride_w  = stride(2);
                otherwise
                    APP_LOG('last_error','Erroneous stride dimensions for %s. Use single value or array of 2 values [s_h s_w]');
            end
            
            [bottom,top] = structure.get_bottom_top_blobs(name);            
            structure.put_to_map(0,name,'POOLING',bottom,top,s);   
            structure.valid_structure = -1;            
        end
        %% PRINT POOLING LAYER
        function print_POOL_layer(structure,fileID,layer)
            formatSpec= ['layers{\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '      pooling_param{\n'...
                         '            pool: %s\n'...
                         '            kernel_h: %d\n'...
                         '            kernel_w: %d\n'...
                         '            pad_h: %d\n'...
                         '            pad_w: %d\n'...
                         '            stride_h: %d\n'...
                         '            stride_w: %d\n'...
                         '      }\n'...
                         '}'];
            hyper_params = layer.hyper_params;
            %print to file
            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,layer.bottom,...
                                    layer.top,upper(hyper_params.method),hyper_params.kernel_h,...
                                    hyper_params.kernel_w,hyper_params.pad_h,hyper_params.pad_w,...
                                    hyper_params.stride_h,hyper_params.stride_w);
        end
        %% ADD INNER PRODUCT LAYER
        function add_IP_layer(structure,name,number_of_output,bias_term,...
                              weight_filler_type,weight_filler_value,weight_lr_mult,...
                              bias_filler_type,bias_filler_value,bias_lr_mult)
            s.num_output          = number_of_output;
            s.weight_filler_type  = weight_filler_type;
            s.weight_filler_value = weight_filler_value;
            s.weight_lr_mult      = weight_lr_mult;
            s.bias_term           = bias_term;
            s.bias_filler_type    = bias_filler_type;
            s.bias_filler_value   = bias_filler_value;
            s.bias_lr_mult        = bias_lr_mult;
            
            [bottom,top] = structure.get_bottom_top_blobs(name);
            structure.put_to_map(1,name,'INNER_PRODUCT',bottom,top,s);      
            structure.valid_structure = -1;            
        end
        %% PRINT INNER PRODUCT LAYER
        function print_IP_layer(structure,fileID,layer)
            bias_weight_filler    = structure.print_weight_bias_term_filler(layer.hyper_params);
            formatSpec= ['layers{\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '      blobs_lr: %1.12f\n'...
                         '      blobs_lr: %1.12f\n'...
                         '      inner_product_param{\n'...
                         '            num_output: %d\n'...
                         '%s'...
                         '      }\n'...
                         '}'];
            hyper_params = layer.hyper_params;
            %print to file
            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,layer.bottom,layer.top,...
                                    hyper_params.weight_lr_mult,hyper_params.bias_lr_mult,...
                                    hyper_params.num_output,bias_weight_filler);
        end
        %% ADD DROPOUT LAYER
        function add_DROPOUT_layer(structure,name,p)
            s.dropout_ratio         = p;
            
            [bottom,~] = structure.get_bottom_top_blobs(name);            
            structure.put_to_map(0,name,'DROPOUT',bottom,bottom,s);
            structure.valid_structure = -1;            
        end
        %% PRINT DROPOUT LAYER
        function print_DROPOUT_layer(structure,fileID,layer)
            formatSpec= ['layers{\n'...
                         '      name: "%s"\n'...
                         '      type: %s\n'...
                         '      bottom: "%s"\n'...
                         '      top: "%s"\n'...
                         '      dropout_param{\n'...
                         '            dropout_ratio: %1.10f\n'...
                         '      }\n'...
                         '}'];
            hyper_params = layer.hyper_params;
            structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.type,...
                                    layer.bottom,layer.top,hyper_params.dropout_ratio);
        end
        %% ADD OUPUT/LOSS LAYER
        function add_OUTPUT_ERROR_layer(structure,type)
            s.type = upper(type);
            name = 'output_or_error';
            [bottom,~] = structure.get_bottom_top_blobs(name);            
            structure.put_to_map(0,name,'OUTPUT_OR_ERROR',bottom,'last',s);
            structure.valid_structure = -1;            
        end
        %% PRINT OUTPUT LAYER
        function print_OUTPUT_layer(structure,fileID,layer)
            switch(layer.hyper_params.type)
                case 'SOFTMAX_LOSS'
                    formatSpec= ['layers{\n'...
                                 '      name: "%s"\n'...
                                 '      type: SOFTMAX\n'...
                                 '      bottom: "%s"\n'...
                                 '      top: "%s"\n'...
                                 '}'];
                    structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.bottom,layer.top);
                case 'CROSS_ENTROPY'
                    formatSpec= ['layers{\n'...
                                 '      name: "%s"\n'...
                                 '      type: SIGMOID\n'...
                                 '      bottom: "%s"\n'...
                                 '      top: "%s"\n'...
                                 '}'];
                    structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.bottom,layer.top);
            end
        end
        %% PRINT ERROR LAYER
        function print_ERROR_layer(structure,fileID,layer)
            switch(layer.hyper_params.type)
                case 'SOFTMAX_LOSS'
                    formatSpec= ['layers{\n'...
                                 '      name: "%s"\n'...
                                 '      type: SOFTMAX_LOSS\n'...
                                 '      bottom: "%s"\n'...
                                 '      bottom: "label"\n'...                                 
                                 '      top: "%s"\n'...
                                 '}'];
                    structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.bottom,layer.top);
                case 'SIGMOID_CROSS_ENTROPY_LOSS'
                    formatSpec= ['layers{\n'...
                                 '      name: "%s"\n'...
                                 '      type: SIGMOID_CROSS_ENTROPY_LOSS\n'...
                                 '      bottom: "%s"\n'...
                                 '      bottom: "label"\n'...                                 
                                 '      top: "%s"\n'...
                                 '}'];
                    structure.WRITE_TO_FILE(fileID,formatSpec,layer.name,layer.bottom,layer.top);
                otherwise
                    APP_LOG('last_error','Unknown Error Layer %s',layer.hyper_params.type);
            end
        end        
        %% PUT TO MAP
        function put_to_map(structure,has_learnable_parameters,name,type,bottom,top,s)
            if(has_learnable_parameters)
                structure.n_layers                                    = structure.n_layers+1;
                switch(type)
                    case 'CONVOLUTION'
                        structure.lr_mult{structure.n_layers,1}=s.weight_lr_mult;
                        structure.lr_mult{structure.n_layers,2}=s.bias_lr_mult;
                    case 'INNER_PRODUCT'
                        structure.lr_mult{structure.n_layers,1}=s.weight_lr_mult;
                        structure.lr_mult{structure.n_layers,2}=s.bias_lr_mult;                        
                    otherwise
                        APP_LOG('last_error','learning rate multiplier handler for %s layer of type %s not present',name,type);
                end
            end            
            structure.counter                                         = structure.counter +1;
            structure.map(structure.counter).has_learnable_parameters = has_learnable_parameters;
            structure.map(structure.counter).name                     = name;
            structure.map(structure.counter).type                     = type;
            structure.map(structure.counter).bottom                   = bottom;
            structure.map(structure.counter).top                      = top;
            structure.map(structure.counter).hyper_params             = s;
            
        end
        %% PRINT TO FILE
        function filepath = create_prototxt(structure,phase)
            name = sprintf('%s_structure.prototxt',phase);
            filepath=fullfile(structure.prototxt_path,name);
            APP_LOG('info','Creating prototxt %s',filepath);
            if(structure.valid_structure~=1)
                APP_LOG('last_error','Not a valid network structure. Validate with .validate_structure()');
            end
            fileID = fopen(filepath,'w');
            structure.WRITE_TO_FILE(fileID,'name: "%s"',phase);
            structure.WRITE_TO_FILE(fileID,'input: "data"');
            switch(phase)
                case 'train'
                        structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.train_batch_size);
                case 'validation'
                        structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.val_batch_size);
                case 'test'
                        structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.test_batch_size);
                otherwise
                    APP_LOG('last_error','Phase "%s" does not exist, use train/validation/test');
            end
            structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.object_depth);
            structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.object_height);
            structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.object_width);
            if(strcmp(phase,'train')==1)
                structure.WRITE_TO_FILE(fileID,'input: "label"');
                structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.train_batch_size);
                structure.WRITE_TO_FILE(fileID,'input_dim: %d',structure.labels_length);
                structure.WRITE_TO_FILE(fileID,'input_dim: 1');
                structure.WRITE_TO_FILE(fileID,'input_dim: 1');
            end
            layer_counter=1;
            for i=1:length(structure.map)
                layer=structure.map(i);                
                if(layer.has_learnable_parameters)
                    structure.WRITE_TO_FILE(fileID,'# --------------------------------layer %d----------------------------------',layer_counter);
                    layer_counter=layer_counter + 1;
                end
                switch(layer.type)
                	case 'CONVOLUTION'
                    	structure.print_CONV_layer(fileID,layer)
                    case 'RELU'
                        structure.print_ACTIV_layer(fileID,layer);                        
                    case 'SIGMOID'
                        structure.print_ACTIV_layer(fileID,layer);                        
                    case 'TANH'
                        structure.print_ACTIV_layer(fileID,layer);
                    case 'LRN'
                        structure.print_LRN_layer(fileID,layer);
                    case 'POOLING'
                        structure.print_POOL_layer(fileID,layer);
                    case 'INNER_PRODUCT'
                        structure.print_IP_layer(fileID,layer);
                    case 'DROPOUT'
                        structure.print_DROPOUT_layer(fileID,layer);
                    case 'OUTPUT_OR_ERROR'
                        switch(phase)
                            case 'train'
                                structure.print_ERROR_layer(fileID,layer);
                            case 'validation'
                                structure.print_OUTPUT_layer(fileID,layer);                                
                            case 'test'                                
                                structure.print_OUTPUT_layer(fileID,layer);
                        end
                end
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
            if structure.object_depth <= 0
                APP_LOG('last_error','Object Depth not set');
            end
            if structure.object_height <= 0
                APP_LOG('last_error','Object Height not set');
            end
            if structure.object_width <= 0
                APP_LOG('last_error','Object Width not set');
            end
            if structure.labels_length <= 0
                APP_LOG('last_error','Labels length not set');
            end
            
            APP_LOG('debug','Outputs And Weights/Bias');
            APP_LOG('debug','###. %15s %6s%7s%6s%10s%5s','LAYER NAME','WIDTH','HEIGHT','DEPTH','WEIGHTS','BIAS');
            output.width = structure.object_width;
            output.height= structure.object_height;            
            output.depth = structure.object_depth;
            APP_LOG('debug','%3d. %15s:[%5d,%5d,%5d]',0,'DATA',output.width,output.height,output.depth);            
            
            total_weights_bias =0;

            layer_counter = 0;
            for i=1:length(structure.map)
                layer=structure.map(i);
                if(layer.has_learnable_parameters)
                    layer_counter=layer_counter+1;
                end
                hyper_params = layer.hyper_params;
                switch (layer.type)
                    case 'CONVOLUTION'                        
                        parameters.weights = hyper_params.kernel_h*hyper_params.kernel_w*output.depth*hyper_params.num_output;

                        if layer.hyper_params.bias_term
                            parameters.bias  =  hyper_params.num_output;
                        else
                            parameters.bias  =  0;
                        end
                        structure.map(i).parameters = parameters;
                        total_weights_bias = total_weights_bias + parameters.weights + parameters.bias;
                        
                        output.width =( output.width - hyper_params.kernel_w + 2*hyper_params.pad_w)/hyper_params.stride_w + 1;
                        output.height=(output.height - hyper_params.kernel_h + 2*hyper_params.pad_h)/hyper_params.stride_h + 1;
                        output.depth = hyper_params.num_output;
                    case 'POOLING'
                        output.width = (output.width - hyper_params.kernel_w + 2*hyper_params.pad_w)/hyper_params.stride_w + 1;
                        output.height=(output.height - hyper_params.kernel_h + 2*hyper_params.pad_h)/hyper_params.stride_h + 1;
                    case 'INNER_PRODUCT'
                        parameters.weights   = output.width*output.height*output.depth*hyper_params.num_output;
                        if layer.hyper_params.bias_term
                            parameters.bias  =  hyper_params.num_output;
                        else
                            parameters.bias  =  0;
                        end
                        structure.map(i).parameters = parameters;
                        total_weights_bias = total_weights_bias + parameters.weights + parameters.bias;
                        
                        output.height = 1;
                        output.width  = 1;
                        output.depth  = hyper_params.num_output;
                end
                if ~isempty(structure.map(i).parameters)
                APP_LOG('debug','%3d. %15s:[%5d,%5d,%5d]%10d%5d',layer_counter,upper(layer.name),output.width,output.height,output.depth,parameters.weights,parameters.bias);
                else
                APP_LOG('debug','%3d. %15s:[%5d,%5d,%5d]',layer_counter,upper(layer.name),output.width,output.height,output.depth);                    
                end
                if ~(output.width==floor(output.width) && output.height==floor(output.height) && output.depth==floor(output.depth))
                    APP_LOG('error','Output feature map of %s layer has non-integer bounds',layer{2});
                    APP_LOG('last_error','Change the configuration considering the documentation');
                end
                structure.map(i).output     = output;
            end
            
            layer_counter = 0;
            for i=1:length(structure.map)
                layer=structure.map(i);
                if(layer.has_learnable_parameters)
                    layer_counter=layer_counter+1;
                    if structure.lr_mult{layer_counter,1} == 0 || structure.lr_mult{layer_counter,2} == 0
                        APP_LOG('debug','Local learning rate multiplier on layer %s is 0. Weights or bias will be treated as constants during training',layer.name);
                    end
                end
            end            
            
            APP_LOG('debug','Structure validity test passed');
            structure.valid_structure=1;
            
            APP_LOG('debug','Estimated size of learn-able parameters: %3.2f MB',((total_weights_bias*4)/1024)/1024);
            total_feature_values=structure.object_height*structure.object_width*structure.object_depth;
            for i=1:length(structure.map)
                layer=structure.map(i);
                if strcmp(layer.bottom,layer.top)==0
                    total_feature_values=total_feature_values + layer.output.height*layer.output.width*layer.output.depth;
                end
            end
            train_feature_values = total_feature_values * structure.train_batch_size;
            APP_LOG('debug','Estimated size of output volumes during training phase: %3.2f MB',((train_feature_values*4)/1024)/1024);
            validation_feature_values = total_feature_values * structure.val_batch_size;
            APP_LOG('debug','Estimated size of output volumes during validation phase: %3.2f MB',((validation_feature_values*4)/1024)/1024);            
            test_feature_values = total_feature_values * structure.test_batch_size;
            APP_LOG('debug','Estimated size of output volumes during test phase: %3.2f MB',((test_feature_values*4)/1024)/1024);
        end
        
        %% WRITE LAYERS TO FILE AND PRINT TO DEBUG SCREEN
        function WRITE_TO_FILE(structure,fileID,fmt,varargin)
            fprintf(fileID,[fmt '\n'],varargin{:});
            %fprintf([fmt '\n'],varargin{:});
            %APP_LOG('debug',fmt,varargin{:});
        end
    end
    
end

