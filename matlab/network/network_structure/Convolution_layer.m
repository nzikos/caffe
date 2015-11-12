classdef Convolution_layer < handle
    %CONVOLUTION_LAYER used to create object handlers for convolution
    %layers
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       num_output,
    %       kernel,
    %       stride,
    %       pad,
    %       bias_term
    %   }
    %
    %   Hints:
    %   1. In case of squared filters a single value can be used under
    %      kernel,stride,pad params.
    %   2. If there is need for different height x width filter's
    %      hyper-parameters a 2-valued array can be used instead.
    %   3. In order to dismiss a layer from the optimization process use
    %      local_multipliers of zero.
    %   5. In order to dismiss weight decay regularization use a local_wd_mult
    %      of zero.
    %
    %   Example input:
    %       {96,[11 11],[4 4],0,true}
    %   Translated as:
    %       96 filters, 11x11  kernels, 4x4 stride, 0x0 padding , use bias
    %
    %   Initialization of weights/bias is happening in another level
    %   In this level, a dummy initialization is forcing caffe to 
    %   initialize itself.
    %
    %   Method process_local_multipliers is expecting a cell in the form
    %   {
    %       {
    %           weight_local_m
    %           weight_local_lr,
    %           weight_local_wd
    %       },
    %       {
    %           bias_local_m    
    %           bias_local_lr,
    %           bias_local_wd
    %       }
    %   }
    %
    %   2nd cell is used only when the use of bias was explicitly set.
    %
    %   Method fill_params is expecting a cell in the form
    %   {
    %       {
    %          type,
    %          params
    %       },
    %       {
    %           type,
    %           params
    %       }
    %   }
    %
    %   2nd cell is used only when the use of bias was explicitly set.
    %
    %   Attributes of this class are:
    %
    %   1.  Bottom:         Bottom blobs' name.
    %   2.  Bottom_size:    Bottom blobs' size.
    %   3.  Top:            Top blobs' name.
    %   4.  Top_size:       Top blobs' size.
    %   5.  Name:           Layer's name.
    %   6.  weights_size:   size of weights
    %   7.  bias_size:      size of bias
    %
    %   8.  num_output:     Number of convolutional kernels.
    %   9.  kernel_h:       Height of convolutional kernels.
    %   10. kernel_w:       Width  of convolutional kernels.
    %   11. stride_h:       Step along height between 2 receptive fields.
    %   12. stride_w:       Step along width  between 2 receptive fields.
    %   13. pad_h:          Padding along height.
    %   14. pad_w:          Padding along width.
    %   15. bias_term:      Use a bias on each convolutional kernel.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   08/11/2015
    %  FOR:    VISION TEAM - AUTH
    
    properties (SetAccess=private)
        bottom;
        bottom_size;
        top;
        top_size;
        name;
        weights_size;
        bias_size;
        
        num_output;
        kernel_h;
        kernel_w;        
        stride_h;
        stride_w;        
        pad_h;
        pad_w;        
        bias_term      = 0;
    end
    
    methods
        function obj = Convolution_layer(bottom,bottom_size,top,name,hyper_params)
            
            % PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            % PROCESS LAYER'S HYPER-PARAMETERS
            if length(hyper_params)~=5
                APP_LOG('last_error','%s: Wrong number of hyper-parameters passed');
            end
            if ~(isnumeric(hyper_params{1}) && floor(hyper_params{1})==hyper_params{1} && hyper_params{1}>0)
                APP_LOG('last_error','%s: Expecting a numerical non-zero integer attribute for num_output',obj.name);                
            end
            obj.num_output        = hyper_params{1};
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
            %bias term
            if hyper_params{5}~=1 && hyper_params{5}~=0
                APP_LOG('last_error','Expecting boolean bias term');
            end
            obj.bias_term         = hyper_params{5};            
            
            % PROCESS LAYER'S OUTPUT SIZE
            obj.top_size(1) = (obj.bottom_size(1) - obj.kernel_h + 2*obj.pad_h)/obj.stride_h + 1;
            obj.top_size(2) = (obj.bottom_size(2) - obj.kernel_w + 2*obj.pad_w)/obj.stride_w + 1;
            obj.top_size(3) = obj.num_output;
                       
            if ~(obj.top_size(1)==floor(obj.top_size(1)) && obj.top_size(2)==floor(obj.top_size(2)) && obj.top_size(3)==floor(obj.top_size(3)))
                APP_LOG('error','%s: Output feature map has non-integer bounds',obj.name);
                APP_LOG('error','Output feature map: [%f %f %f]',obj.top_size(1),obj.top_size(2),obj.top_size(3));
                APP_LOG('last_error','Change the configuration considering the documentation');
            end
            if ~all(sign(obj.top_size))
                APP_LOG('error','Output feature map: [%f %f %f]',obj.top_size(1),obj.top_size(2),obj.top_size(3));                
                APP_LOG('last_error','Negative/zero output dimensions detected on %s. Consider, a smaller kernel or greater pad',obj.name);
            end
            
            % PROCESS LAYER'S LEARNABLE-PARAMS SIZE
            obj.weights_size = [obj.kernel_h,obj.kernel_w,obj.bottom_size(3),obj.num_output];
            if obj.bias_term
                obj.bias_size    = [1,1,1,obj.num_output];
            end
            
            % PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);            
            APP_LOG('debug','num_output: %d'           ,obj.num_output);
            APP_LOG('debug','kernel_h: %d'             ,obj.kernel_h);
            APP_LOG('debug','kernel_w: %d'             ,obj.kernel_w);
            APP_LOG('debug','stride_h: %d'             ,obj.stride_h);
            APP_LOG('debug','stride_w: %d'             ,obj.stride_w);
            APP_LOG('debug','pad_h: %d'                ,obj.pad_h);
            APP_LOG('debug','pad_w: %d'                ,obj.pad_w);
            APP_LOG('debug','bias_term: %d'            ,obj.bias_term);
            APP_LOG('debug','');
        end

        %% PROCESS LAYER'S LOCAL MULTIPLIERS
        function [m_mult,lr_mult,wd_mult] = process_local_multipliers(obj,hyper_params)
            
            m_mult  = ones(1,2);
            lr_mult = ones(1,2);
            wd_mult = ones(1,2);
            if length(hyper_params)~=1+obj.bias_term
                APP_LOG('last_error','%s: Expecting a cell in the form {[w_lr_mult,w_wd_mult],[b_lr_mult,b_wd_mult]}',obj.name);
            end
            if length(hyper_params{1})~=3
                APP_LOG('last_error','%s: Erroneous arguments passed to local weight multipliers',obj.name);
            end
            if ~(isnumeric(hyper_params{1}) && all(hyper_params{1}>=0))
                APP_LOG('last_error','%s: Expecting numeric positive (or zero) local weight multipliers',obj.name);
            end
            %local weight multipliers            
            m_mult(1,1) = hyper_params{1}(1);
            lr_mult(1,1)= hyper_params{1}(2);
            wd_mult(1,1)= hyper_params{1}(3);
            if lr_mult(1,1)==0 && wd_mult(1,1)>0
                APP_LOG('warning','%s: Weights may converge to zero',obj.name);      %depends on optimization method
            end
            
            if obj.bias_term
                if length(hyper_params{2})~=3
                    APP_LOG('last_error','%s: Erroneous arguments passed to local bias multipliers',obj.name);
                end
                if ~(isnumeric(hyper_params{2}) && all(hyper_params{2}>=0))
                    APP_LOG('last_error','%s: Expecting numeric positive (or zero) local bias multipliers',obj.name);
                end
                %local bias multipliers                
                m_mult(1,2)  = hyper_params{2}(1);
                lr_mult(1,2) = hyper_params{2}(2);
                wd_mult(1,2) = hyper_params{2}(3);
                if lr_mult(1,2)==0 && wd_mult(1,2)>0
                    APP_LOG('warning','%s: Bias may converge to zero',obj.name);  %depends on optimization method
                end
            else
                m_mult(1,2)   = 0;
                lr_mult(1,2)  = 0;
                wd_mult(1,2)  = 0;
                APP_LOG('debug','%s: Ignoring bias local multipliers',obj.name);
            end
            
            APP_LOG('debug','weight_m_mult: %1.12f'    ,m_mult(1,1));
            APP_LOG('debug','weight_lr_mult: %1.12f'   ,lr_mult(1,1));
            APP_LOG('debug','weight_wd_mult: %1.12f'   ,wd_mult(1,1));
            if obj.bias_term
                APP_LOG('debug','bias_m_mult: %1.12f'  ,m_mult(1,2));
                APP_LOG('debug','bias_lr_mult: %1.12f' ,lr_mult(1,2));
                APP_LOG('debug','bias_wd_mult: %1.12f' ,wd_mult(1,2));
            end
            APP_LOG('debug','');            
        end
        
        %% INIT LAYER'S PARAMETERS
        function out = fill_params(obj,fill_params)
            %init tmp_filler_function
            if length(fill_params)~=1+obj.bias_term
                APP_LOG('last_error','%s: Expecting a cell in the form {{''type'',params},{''type'',params}}',obj.name);
            end
            tmp_filler   = FILLER();
            out{1,1}     = tmp_filler.fill(obj.weights_size,fill_params{1});
            if obj.bias_term
                out{1,2} = tmp_filler.fill(obj.bias_size,fill_params{2});
            end
            APP_LOG('debug','Weights size: [%d, %d, %d, %d]',obj.weights_size(1),obj.weights_size(2),obj.weights_size(3),obj.weights_size(4));
            APP_LOG('debug','Weights initialization method: %s',fill_params{1}{1});
            if obj.bias_term
                APP_LOG('debug','Bias size: [%d, %d, %d, %d]',obj.bias_size(1),obj.bias_size(2),obj.bias_size(3),obj.bias_size(4));
                APP_LOG('debug','Bias initialization method: %s',fill_params{2}{1});
            end
            APP_LOG('debug','');
        end
        
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                             'name: "%s"\n\t'...
                             'type: "Convolution"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             'convolution_param {\n\t\t'...
                                 'num_output: %d\n\t\t'...
                                 'kernel_h: %d\n\t\t'...
                                 'kernel_w: %d\n\t\t'...
                                 'stride_h: %d\n\t\t'...
                                 'stride_w: %d\n\t\t'...
                                 'pad_h: %d\n\t\t'...
                                 'pad_w: %d\n\t\t'...
                                 'bias_term: %s\n\t'...
                             '}\n'...
                         '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,obj.num_output,...
                                          obj.kernel_h,obj.kernel_w,obj.stride_h,obj.stride_w,...
                                          obj.pad_h,obj.pad_w,bool2str(obj.bias_term));
            else
                APP_LOG('last_error','No filepath to print prototxt supplied');
            end
%             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,obj.num_output,...
%                                           obj.kernel_h,obj.kernel_w,obj.stride_h,obj.stride_w,...
%                                           obj.pad_h,obj.pad_w);
        end
    end
    
end


