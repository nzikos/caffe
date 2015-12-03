classdef MVN_layer < handle
    %MVN_LAYER used to create object handlers for MVN layers.
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       normalize_variance,
    %       across_channels
    %   }    
    %   Example input:
    %       {false,true}
    %   Translated as:
    %       normalize mean taking into account the whole volume
    %
    %   Attributes of this class are:
    %
    %   1. Bottom:         Bottom blobs' name.
    %   2. Bottom_size:    Bottom blobs' size.
    %   3. Top:            Top blobs' name.
    %   4. Top_size:       Top blobs' size.
    %   5. Name:           Layer's name.
    %
    %   6. normalize_variance:  Whether normalize variance or not.
    %   7. across_channels:     Whether to use statistics along all channels or
    %                           not.
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
        
        normalize_variance;
        across_channels;
    end
    
    methods
        function obj = MVN_layer(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            if length(hyper_params)~=2
                APP_LOG('last_error','%s: Expecting two attributes {normalize_variance,across_channels}',obj.name);
            end
            %normalize variance
            if ~(hyper_params{1}==false || hyper_params{1}==true)
                APP_LOG('last_error','%s: Normalize variance attribute is expecting a true/false value',obj.name);                
            end
            obj.normalize_variance= hyper_params{1};
            
            %across-channels
            if ~(hyper_params{2}==false || hyper_params{2}==true)
                APP_LOG('last_error','%s: Across channels attribute is expecting a true/false value',obj.name);
            end
            obj.across_channels   = hyper_params{2};            
            
            %% PROCESS LAYER'S OUTPUT SIZE
            obj.top_size       = bottom_size;

            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','normalize_variance: %s'   ,bool2str(obj.normalize_variance));
            APP_LOG('debug','across_channels: %s'      ,bool2str(obj.across_channels));
            APP_LOG('debug','');
        end
        
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                             'name: "%s"\n\t'...
                             'type: "MVN"\n\t'...
                             'bottom: "%s"\n\t'...
                             'top: "%s"\n\t'...
                             'mvn_param {\n\t\t'...
                                 'normalize_variance: %s\n\t\t'...
                                 'across_channels: %s\n\t'...
                             '}\n'...
                         '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,...
                                               bool2str(obj.normalize_variance),bool2str(obj.across_channels));
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
%             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,...
%                                                bool2str(obj.normalize_variance),bool2str(obj.across_channels));
        end                
    end
    
end

