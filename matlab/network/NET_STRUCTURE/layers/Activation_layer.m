classdef Activation_layer < handle
    %ACTIVATION_LAYER used to create object handlers for activation layers
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Supported activation units are:
    %       1. ReLU
    %       2. Sigmoid
    %       3. Tanh
    %
    %   In case of ReLU, constructor is expecting a cell input in the form
    %   {
    %      negative_slope
    %   }    
    %
    %   Attributes of this class are:
    %
    %   1. Bottom:         Bottom blobs' name.
    %   2. Bottom_size:    Bottom blobs' size.
    %   3. Top:            Top blobs' name.
    %   4. Top_size:       Top blobs' size.
    %   5. Name:           Layer's name.
    %
    %   6. type:           ReLU / Sigmoid / Tanh
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
        
        type;
        negative_slope = 0;
    end
    
    methods
        function obj = Activation_layer(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            obj.type           = hyper_params{1};

            switch obj.type
                case 'Sigmoid'
                case 'Tanh'
                case 'ReLU'
                    if length(hyper_params)==2
                        if ~isnumeric(hyper_params{2})
                            APP_LOG('last_error','%s: Expecting numeric and positive negative_slope as 2nd ReLU param',obj.name);
                        end
                        if hyper_params{2}<0
                            APP_LOG('last_error','%s: Leaky ReLU is not supposed to accept negative slopes < 0',obj.name);                            
                        end
                        obj.negative_slope = hyper_params{2};   
                    end
                otherwise
                    APP_LOG('last_error','%s: Unknown Activation method %s. Use Sigmoid/Tanh/ReLU',obj.name,obj.type);
            end
            
            %% PROCESS LAYER'S OUTPUT SIZE
            obj.top_size       = bottom_size;

            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','type: %s'                 ,obj.type);
            switch obj.type
                case 'ReLU'
                    APP_LOG('debug','negative_slope: %1.12f',obj.negative_slope);
            end
            APP_LOG('debug','');                        
        end
        
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                             'name: "%s"\n\t'...
                             'type: "%s"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             '%s'...
                         '}\n'];
            print_params=[];
            switch obj.type
                case 'ReLU'
                    print_params = sprintf('relu_param {\n\t\tnegative_slope: %1.12f\n\t}\n',obj.negative_slope);
            end
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.type,obj.bottom,obj.top,print_params);
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
        end
    end
    
end

