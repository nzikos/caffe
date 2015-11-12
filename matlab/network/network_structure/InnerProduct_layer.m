classdef InnerProduct_layer < handle
    %INNERPRODUCT_LAYER used to create object handlers for InnerProduct
    %layers
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       num_output,
    %       bias_term
    %   }
    %
    %   Example input:
    %       {4096,false}
    %   Translated as:
    %       Use 4096 ubiased neurons.
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
    %   Attributes of this class are:
    %
    %   1. Bottom:         Bottom blobs' name.
    %   2. Bottom_size:    Bottom blobs' size.
    %   3. Top:            Top blobs' name.
    %   4. Top_size:       Top blobs' size.
    %   5. Name:           Layer's name.
    %   6. weights_size:   Size of weights
    %   7. bias_size:      Size of bias
    %
    %   8. num_output:     Number of neurons.
    %   9. bias_term:      Whether to use bias on each neuron or not
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
        bias_term      = 0;        
    end
    
    methods
        function obj = InnerProduct_layer(bottom,bottom_size,top,name,hyper_params)
            
            % PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            % PROCESS LAYER'S HYPER-PARAMETERS
            if length(hyper_params)~=2
                APP_LOG('last_error','%s: Wrong number of hyper-parameters passed');                
            end
            if ~isnumeric(hyper_params{1}) || hyper_params{1}==0
                APP_LOG('last_error','%s: Cant use non_numeric/zero neurons',name);
            end
            if ~(hyper_params{2}==1 || hyper_params{2}==0)
                APP_LOG('last_error','Expecting boolean bias term');
            end
            obj.num_output        = hyper_params{1};
            obj.bias_term         = hyper_params{2};
            
            % PROCESS LAYER'S OUTPUT SIZE       
            obj.top_size    = [1,1,obj.num_output];
            
            if ~(obj.top_size(1)==floor(obj.top_size(1)) && obj.top_size(2)==floor(obj.top_size(2)) && obj.top_size(3)==floor(obj.top_size(3)))
                APP_LOG('error','%s: Output feature map has non-integer bounds',obj.name);
                APP_LOG('error','[%f %f %f]',obj.top_size(1),obj.top_size(2),obj.top_size(3));
                APP_LOG('last_error','Use an integer amount of neurons!');
            end            
            
            % PROCESS LAYER'S LEARNABLE-PARAMS SIZE
            obj.weights_size     = [1,1,obj.bottom_size(1)*obj.bottom_size(2)*obj.bottom_size(3),obj.num_output];
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
                             'type: "InnerProduct"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             'inner_product_param {\n\t\t'...
                                 'num_output: %d\n\t\t'... 
                                 'bias_term: %s\n\t'...
                             '}\n'...
                         '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,obj.num_output,bool2str(obj.bias_term));
            else
                APP_LOG('last_error','No filepath to print prototxt supplied');
            end
%             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,obj.num_output);
        end
    end
end

