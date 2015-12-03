classdef caffe_filler < handle
    %% DEPRECATED 
    %
    %   INITITIALIZATION OF LEARNABLE PARAMETERS IS TAKING PLACE INSIDE MATLAB
    %   see FILLER.m
    %
    %CAFFE_FILLER This function is used to store the filler parameters of
    %layers with learnable-parameters.
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Available filler methods along with their parameters are:
    %    1. constant -> value
    %    2. uniform  -> [min,max]
    %    3. gaussian -> [mean,std]
    %    4. xavier
    %    5. positive_unitball
    %
    %   This initialization method is deprecated and thus unused.
    %   Layers parameters are set inside matlab and passed to caffe
    %   before training begins.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   06/11/2015
    %  FOR:    VISION TEAM - AUTH
    
    properties (SetAccess=private)
        filler_type;
        value;
    end
    
    methods
        function obj = caffe_filler(params)
            obj.set_filler(params);
        end
        
        function set_filler(obj,params)
            params{1}=lower(params{1});
            
            obj.filler_type = params{1};            
            
            switch params{1}
                case 'constant'
                    if length(params)==2
                        if length(params{2})==1
                            obj.value = params{2};
                        else
                            APP_LOG('last_error','Expecting a single value for constant filler type');
                        end
                    else
                        APP_LOG('last_error','Expecting a single value for constant filler type');
                    end
                case 'uniform'
                    if length(params)==2
                        if length(params{2})==2
                            obj.value = params{2};
                        else
                            APP_LOG('last_error','Expecting two values for uniform filler type [min,max]');
                        end
                    else
                        APP_LOG('last_error','Expecting two values for uniform filler type [min,max]');
                    end
                case 'gaussian'
                    if length(params)==2
                        if length(params{2})==2
                            obj.value = params{2};
                        else
                            APP_LOG('last_error','Expecting two values for gaussian filler type [mean,std]');
                        end
                    else
                        APP_LOG('last_error','Expecting two values for gaussian filler type [mean,std]');
                    end
                case 'xavier'
                    if length(params)==2
                        APP_LOG('warning','Xavier filler is not expecting any parameters. Ignoring value(s)');
                    end
                case 'positive_unit_ball'
                    if length(params)==2
                        APP_LOG('warning','Positive unit ball filler is not expecting any parameters. Ignoring value(s)');
                    end
                otherwise
                    APP_LOG('last_error',['Unsupported initialization method %s\n'...
                                         'Use: constant/uniform/gaussian/xavier/positive_unitball'],type);

            end
        end
        
        function out = get_filler(obj)
            switch(obj.filler_type)
                case 'gaussian'
                    formatSpec =['type: "gaussian"\n\t\t\t'...
                                 'mean: %1.10f\n\t\t\t'...
                                 'std: %1.10f\n\t\t'];
                    out = sprintf(formatSpec,obj.value(1),obj.value(2));
                case 'constant'
                    formatSpec =['type: "constant"\n\t\t\t'...
                                 'value: %1.10f\n\t\t'];
                    out = sprintf(formatSpec,obj.value);
                case 'uniform'
                    formatSpec =['type: "uniform"\n\t\t\t'...
                                 'min: %1.10f\n\t\t\t'...
                                 'max: %1.10f\n\t\t'];
                    out = sprintf(formatSpec,obj.value(1),obj.value(2));
                case 'xavier'
                    formatSpec ='type: "xavier"\n\t\t';
                    out = sprintf(formatSpec);
                case 'positive_unitball'
                    formatSpec ='type: "positive_unitball"\n\t\t';
                    out = sprintf(formatSpec);
            end
        end
        
    end
    
end

