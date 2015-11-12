classdef PReLU_layer < handle
    %defines a structure which contains the parameters of
    %the PReLU layer.
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %      channel_shared
    %      local_multipliers = [local_m_mult, local_lr_mult , local_wd_mult]
    %   }
    %
    %   Hints:
    %   1. In case no local_multipliers are supplied, they are assumed to be [1 1 0].
    %   2. In case local_multipliers are zero, layer is deprecated to Leaky Relu
    %   3. In case local_multipliers are zero and initial_value is zero,
    %      layer is deprecated to simple ReLU.
    %   4. In case channel_shared is true there is a single parameter for every
    %      input. Otherwise, there is a single parameter per Depth.
    %   5. If it is used after an inner product layer with channel_shared=false
    %      then there is a single parameter for each ip layer's output.
    %
    %   Example input {false,[1 1 0]}
    %       Recommended usage from the paper which introduced it
    %       (http://arxiv.org/pdf/1502.01852v1.pdf)
    %   Note that, during tests, this type of initialization was not 
    %   allowing our network to converge leading the PReLU's parameters to 
    %   oscillation. In order to achieve convergence with PReLU we had to 
    %   either shrink local_lr to 0.1 or local_m to 0.1. More tests are 
    %   needed to clarify the usage of this layer.
    %   
    %   Attributes of this class are:
    %
    %   1. Bottom:         Bottom blobs' name.
    %   2. Bottom_size:    Bottom blobs' size.
    %   3. Top:            Top blobs' name.
    %   4. Top_size:       Top blobs' size.
    %   5. Name:           Layer's name.
    %   6. weights_size:   Size of learnable parameters
    %
    %   7. channel_shared: Whether the negative slope parameter is shared
    %                      along channels (depth).
    %
    %   Method process_local_multipliers is expecting a cell in the form
    %   {
    %      weight_local_m,
    %      weight_local_lr,
    %      weight_local_wd
    %   }
    %
    %   Method fill_params is expecting a cell in the form
    %   {
    %       type,
    %       params
    %	}
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   08/11/2015
    %  FOR:    VISION TEAM - AUTH
    
    properties (SetAccess = private)
        bottom;
        bottom_size;
        top;
        top_size;
        name;
        weights_size;
        
        channel_shared;
    end
    
    methods
        function obj = PReLU_layer(bottom,bottom_size,top,name,hyper_params)
            
            % PROCESS LAYER'S CONNECTIONS            
            obj.bottom      = bottom;
            obj.bottom_size = bottom_size;
            obj.top         = top;
            obj.name        = name;

            % PROCESS LAYER'S HYPER-PARAMETERS
            obj.channel_shared = hyper_params{1};

            % PROCESS LAYER'S OUTPUT SIZE
            obj.top_size    = obj.bottom_size;
            
            % PROCESS LAYER'S LEARNABLE PARAMS SIZE
            if obj.channel_shared
                obj.weights_size = [1,1,1,1];
            else
                obj.weights_size = [1,1,1,obj.bottom_size(3)];
            end
            
            % PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','channel_shared: %s'       ,bool2str(obj.channel_shared));
        end
        
        %% PROCESS LAYER'S LOCAL MULTIPLIERS            
        function [m_mult,lr_mult,wd_mult] = process_local_multipliers(obj,hyper_params)
            if length(hyper_params)~=1 || length(hyper_params{1})~=3
                APP_LOG('last_error','%s: Expecting params as {[weight_m_mult, weight_lr_mult, weight_wd_mult]}',obj.name);
            end
            if ~(isnumeric(hyper_params{1}) && all(hyper_params{1}>=0))
                APP_LOG('last_error','%s: Expecting numerical positive (or zero) local multipliers');
            end
            m_mult =hyper_params{1}(1);
            lr_mult=hyper_params{1}(2);
            wd_mult=hyper_params{1}(3);
            if lr_mult==0 && wd_mult>0
               APP_LOG('warning','%s: Parameter may converge to zero');
            end
            APP_LOG('debug','weight_m_mult: %d'        ,m_mult);
            APP_LOG('debug','weight_lr_mult: %d'       ,lr_mult);
            APP_LOG('debug','weight_wd_mult: %d'       ,wd_mult);    
        end
        
        %% INIT LAYER'S PARAMETERS
        function out = fill_params(obj,fill_params)
            %init tmp_filler_function
            tmp_filler   = FILLER();
            out{1,1}     = tmp_filler.fill(obj.weights_size,fill_params);
            APP_LOG('debug','Weights size: [%d, %d, %d, %d]',obj.weights_size(1),obj.weights_size(2),obj.weights_size(3),obj.weights_size(4));
            APP_LOG('debug','Weights initialization method: %s',fill_params{1});
            APP_LOG('debug','');
        end
        
        %% PRINT LAYER ON PROTOTXT        
        function print_layer(obj,varargin)
                       
            formatSpec = ['layer {\n\t'...
                             'name: "%s"\n\t'...
                             'type: "PReLU"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             'prelu_param {\n\t\t'...
                                 'channel_shared: %s\n\t'...
                             '}\n'...
                         '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,bool2str(obj.channel_shared));
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
%             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,bool2str(obj.channel_shared));
        end
    end
    
end

