classdef pooling_layer
    %POOLING_LAYER used to create object handlers for pooling
    %layers
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       pool,
    %       kernel,
    %       stride,
    %       pad,
    %   }    
    %   Example input:
    %       {MAX,3,2,0}
    %   Translated as:
    %       MAX-pooling, 3x3 kernel, 2x2 stride, 0x0 padding
    %
    %   Pooling layers inherited from caffe contain no learnable-parameters
    %   Available Pooling layers are MAX,AVE,STOCHASTIC
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   08/11/2015
    %  FOR:    VISION TEAM - AUTH    
    properties
        bottom;
        bottom_size;
        top;
        top_size;
        name;
        
        pool
        kernel_h;
        kernel_w;        
        stride_h;
        stride_w;        
        pad_h;
        pad_w;        
    end
    
    methods
        function obj = pooling_layer(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            obj.pool           = upper(hyper_params{1});
            
            switch obj.pool
                case 'MAX'
                case 'AVE'
                case 'STOCHASTIC'
                otherwise
                    APP_LOG('last_error','%s: Unknown pooling %s. Use MAX/AVE/STOCHASTIC',obj.name);
            end
            
            %Kernel
            if ~(isnumeric(hyper_params{2}) && all(hyper_params{2}>0))
                APP_LOG('last_error','%s: Expecting a numerical positive kernel',obj.name);                
            end
            switch length(hyper_params{2})
                case 1
                    obj.kernel_h  = hyper_params{2};
                    obj.kernel_w  = hyper_params{2};
                case 2
                    obj.kernel_h  = hyper_params{2}(1);
                    obj.kernel_w  = hyper_params{2}(2);
                otherwise
                    APP_LOG('last_error','%s: Erroneous kernel dimensions. Use single value or array of 2 values [k_h k_w]',obj.name);
            end
            %Stride
            if ~(isnumeric(hyper_params{3}) && all(hyper_params{3}>0))
                APP_LOG('last_error','%s: Expecting a numerical positive stride',obj.name);                
            end
            switch length(hyper_params{3})
                case 1
                    obj.stride_h  = hyper_params{3};
                    obj.stride_w  = hyper_params{3};
                case 2
                    obj.stride_h  = hyper_params{3}(1);
                    obj.stride_w  = hyper_params{3}(2);
                otherwise
                    APP_LOG('last_error','%s: Erroneous stride dimensions. Use single value or array of 2 values [s_h s_w]',obj.name);
            end
            %Pad
            if ~(isnumeric(hyper_params{4}) && all(hyper_params{4}>=0))
                APP_LOG('last_error','%s: Expecting a numerical positive or zero pad',obj.name);
            end            
            switch length(hyper_params{4})
                case 1
                    obj.pad_h     = hyper_params{4};
                    obj.pad_w     = hyper_params{4};
                case 2
                    obj.pad_h     = hyper_params{4}(1);
                    obj.pad_w     = hyper_params{4}(2);
                otherwise
                    APP_LOG('last_error','%s: Erroneous padding dimensions. Use single value or array of 2 values [p_h p_w]',obj.name);
            end
            
            %% PROCESS LAYER'S OUTPUT SIZE
            obj.top_size(1) = (obj.bottom_size(1) - obj.kernel_h + 2*obj.pad_h)/obj.stride_h + 1;
            obj.top_size(2) = (obj.bottom_size(2) - obj.kernel_w + 2*obj.pad_w)/obj.stride_w + 1;
            obj.top_size(3) = obj.bottom_size(3);

            if ~(obj.top_size(1)==floor(obj.top_size(1)) && obj.top_size(2)==floor(obj.top_size(2)) && obj.top_size(3)==floor(obj.top_size(3)))
                APP_LOG('error','%s: Output feature map has non-integer bounds',obj.name);
                APP_LOG('error','Output feature map: [%f %f %f]',obj.top_size(1),obj.top_size(2),obj.top_size(3));
                APP_LOG('last_error','Change the configuration considering the documentation');
            end
            if ~all(sign(obj.top_size))
                APP_LOG('error','Output feature map: [%d %d %d]',obj.top_size(1),obj.top_size(2),obj.top_size(3));                
                APP_LOG('last_error','Negative/zero output dimensions detected on %s. Consider, a smaller kernel or greater pad',obj.name);
            end

            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','pool: %s'                 ,obj.pool);
            APP_LOG('debug','kernel_h: %d'             ,obj.kernel_h);
            APP_LOG('debug','kernel_w: %d'             ,obj.kernel_w);
            APP_LOG('debug','stride_h: %d'             ,obj.stride_h);
            APP_LOG('debug','stride_w: %d'             ,obj.stride_w);
            APP_LOG('debug','pad_h: %d'                ,obj.pad_h);
            APP_LOG('debug','pad_w: %d'                ,obj.pad_w);
            APP_LOG('debug','');
        end
        
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                             'name: "%s"\n\t'...
                             'type: "Pooling"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             'pooling_param {\n\t\t'...
                                 'pool: %s\n\t\t'...
                                 'kernel_h: %d\n\t\t'...
                                 'kernel_w: %d\n\t\t'...
                                 'stride_h: %d\n\t\t'...
                                 'stride_w: %d\n\t\t'...
                                 'pad_h: %d\n\t\t'...
                                 'pad_w: %d\n\t'...                                 
                             '}\n'...
                         '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,obj.pool,...
                                          obj.kernel_h,obj.kernel_w,obj.stride_h,obj.stride_w,...
                                          obj.pad_h,obj.pad_w);
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
%             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,obj.pool,obj.kernel_h,...
%                                        obj.kernel_w,obj.stride_h,obj.stride_w,obj.pad_h,obj.pad_w);
        end        
    end
end

