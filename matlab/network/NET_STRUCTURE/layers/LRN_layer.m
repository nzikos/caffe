classdef LRN_layer < handle
    %LRN_LAYER used to create object handlers for LRN layers.
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       norm_region
    %       local_size
    %       alpha
    %       beta
    %   }    
    %   Example input:
    %       {'ACROSS_CHANNELS',5,0.0001,0.75}
    %   Translated as:
    %       normalize across channels with k=5 α=0.0001 β=0.75
    %       Which is also the default initialization and the type of
    %       initialization suggested from the paper which introduced this
    %       layer:
    %       http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    %
    %   Attributes of this class are:
    %
    %   1. Bottom:         Bottom blobs' name.
    %   2. Bottom_size:    Bottom blobs' size.
    %   3. Top:            Top blobs' name.
    %   4. Top_size:       Top blobs' size.
    %   5. Name:           Layer's name.
    %
    %   6. norm_region:    Across_channels / within_channel
    %   7. local_size:     coefficient k
    %   8. alpha:          coefficient α
    %   9. beta:           coefficient β
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
        
        norm_region;
        local_size;
        alpha;
        beta;
    end
    
    methods
        function obj = LRN_layer(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            if isempty(hyper_params)
                obj.norm_region = 'ACROSS_CHANNELS';
                obj.local_size  = 5;
                obj.alpha       = 0.0001;
                obj.beta        = 0.75;
            else
            	if length(hyper_params)~=4
                     APP_LOG('last_error','%s: Erroneous number of parameters. Use {norm_region,local_size,alpha,beta}',obj.name);
            	end
                %norm region
                obj.norm_region           = upper(hyper_params{1});
                if isnumeric(obj.norm_region)
                    APP_LOG('last_error','%s: Erroneous argument passed to norm_region',obj.name);
                end
                switch(obj.norm_region)
                    case 'ACROSS_CHANNELS'
                    case 'WITHIN_CHANNEL'
                    otherwise
                        APP_LOG('last_error','%s: Unknown method',obj.norm_region);
                end
                
                %local-size
                if length(hyper_params{2})~=1
                    APP_LOG('last_error','%s: Erroneous argument passed to local_size. Use a single value isntead',obj.name);                    
                end
                if ~(isnumeric(hyper_params{2}) && (hyper_params{2}>0) && floor(hyper_params{2})==hyper_params{2})
                    APP_LOG('last_error','%s: Use positive,non zero integer values to declare the local_size',obj.name);                    
                end
                obj.local_size            = hyper_params{2};
                
                %alpha
                if length(hyper_params{3})~=1
                    APP_LOG('last_error','%s: Erroneous argument passed to alpha. Use a single value isntead',obj.name);                    
                end
                if ~isnumeric(hyper_params{3})
                    APP_LOG('last_error','%s: Erroneous argument passed to alpha. Use a single numeric value isntead',obj.name);                    
                end
                obj.alpha                 = hyper_params{3};
                
                %beta
                if length(hyper_params{4})~=1
                    APP_LOG('last_error','Erroneous argument passed to beta of %s layer. Use a single value isntead',obj.name);                    
                end
                if ~isnumeric(hyper_params{4})
                    APP_LOG('last_error','%s: Erroneous argument passed to beta. Use a single numeric value isntead',obj.name);                    
                end
                obj.beta                  = hyper_params{4};
            end
            
            %% PROCESS LAYER'S OUTPUT SIZE
            obj.top_size       = obj.bottom_size;
            
            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                ,obj.name);
            APP_LOG('debug','bottom: %s'               ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]',obj.bottom_size);
            APP_LOG('debug','top: %s'                  ,obj.top);
            APP_LOG('debug','top_size: [%d, %d, %d]'   ,obj.top_size);
            APP_LOG('debug','norm_region: %s'          ,obj.norm_region);
            APP_LOG('debug','local_size: %d'           ,obj.local_size);
            APP_LOG('debug','alpha: %1.12f'            ,obj.alpha);
            APP_LOG('debug','beta: %1.12f'             ,obj.beta);
            APP_LOG('debug','');
        end
        
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            formatSpec = ['layer {\n\t'...
                'name: "%s"\n\t'...
                'type: "LRN"\n\t'...
                'bottom: "%s"\n\t'...
                'top: "%s"\n\t'...
                'lrn_param {\n\t\t'...
                'norm_region: %s\n\t\t'...
                'local_size: %d\n\t\t'...
                'alpha: %1.12f\n\t\t'...
                'beta: %1.12f\n\t'...
                '}\n'...
                '}\n'];
            if ~isempty(varargin{1})
                fprintf(varargin{1},formatSpec,obj.name,obj.bottom,obj.top,...
                    obj.norm_region,obj.local_size,obj.alpha,obj.beta);
            else
                APP_LOG('warning','No filepath to print prototxt supplied');
            end
            %             APP_LOG('debug',formatSpec,obj.name,obj.bottom,obj.top,...
            %                                                obj.norm_region,obj.local_size,obj.alpha,obj.beta);
        end
    end
    
end
