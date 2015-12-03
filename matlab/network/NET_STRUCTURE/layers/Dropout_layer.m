classdef Dropout_layer
    %DROPOUT_LAYER used to create object handlers for dropout layers
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %      dropout_ratio
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
    %   6. dropout_ratio:  The ratio of disabled over total neurons per
    %                      training iteration.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   09/11/2015
    %  FOR:    VISION TEAM - AUTH
    properties
        bottom;
        bottom_size;
        top;
        top_size;
        name;
        
        dropout_ratio;        
    end
    
    methods
        function obj = Dropout_layer(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            if ~isempty(hyper_params)
                if ~(hyper_params{1}>=0 && hyper_params{1}<1)
                    APP_LOG('last_error','%s: Erroneous dropout_ratio %1.12f. Use values between [0,1)',obj.name,hyper_params{2});                    
                end
                    obj.dropout_ratio     = hyper_params{1};
            else
                obj.dropout_ratio = 0.5;
                APP_LOG('warning','%s: Empty dropout_ratio supplied. Using default 0.5',obj.name);
            end
            
            %% PROCESS LAYER'S OUTPUT SIZE       
            obj.top_size    = obj.bottom_size;
            
            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','dropout_ratio: %1.12f'    ,obj.dropout_ratio);            
        end
        
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                'name: "%s"\n\t'...
                'type: "Dropout"\n\t'...
                'bottom: "%s"\n\t'...
                'top: "%s"\n\t'...
                'dropout_param {\n\t\t'...
                'dropout_ratio: %1.12f\n\t'...
                '}\n'...
                '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,[obj.name '_1'],obj.bottom,obj.top,obj.dropout_ratio);
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
        end
    end    
end

