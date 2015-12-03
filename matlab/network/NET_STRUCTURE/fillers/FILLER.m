classdef FILLER
    %FILLER This class contains the functions used to initialize the 
    %       learn-able parameters of the CNN.
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Available filler types along with their parameters are:
    %    1. constant -> value
    %    2. uniform  -> [min,max]
    %    3. gaussian -> [mean,std]
    %
    %   These types are a best try to imitate the way CAFFE is
    %   initializing it's own weights/bias.
    %
    %	Method fill is expecting the dimensions of the output volume
    %   as HxWxDxN. The parameters of the initilazation proccess which
    %	are passed as a cell input in the form
    %	{
    %		type,
    %		array of type specific parameters
    %	}
    %
    %	Output volume is a 4D array with permuted width and height as
    %	WxHxDxN (column-major) -> NxDxHxW (row-major)
    %	
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   08/11/2015
    %  FOR:    VISION TEAM - AUTH    
    
    properties
        s;
    end
    
    methods
        function obj = FILLER()
%            obj.s = RandStream('mt19937ar','Seed',1);
        end
        
        function out = fill(obj,dims,params)
            if ~isnumeric(params{1})
                switch(params{1})
                    case 'uniform'
                        if isnumeric(params{2}) && length(params{2})==2
                            min_value = min(params{2});
                            max_value = max(params{2});
                            for i=dims(4):-1:1
                                out(:,:,:,i) = min_value + (max_value - min_value)*rand(dims(1:3));
                            end
                        else
                            APP_LOG('last_error','Erroneous parameters on uniform filler. Use [min,max]');
                        end
                    case 'constant'
                        if isnumeric(params{2}) && length(params{2})==1
                            value = params{2};
                            out(1:dims(1),1:dims(2),1:dims(3),1:dims(4))=value;
                        else
                            APP_LOG('last_error','Erroneous parameters on constant filler. Use [value]');
                        end
                    case 'gaussian'
                        if isnumeric(params{2}) && length(params{2})==2
                            mean = params{2}(1);
                            std  = params{2}(2);
                            for i=dims(4):-1:1
                                out(:,:,:,i)  = mean + std.*randn(dims(1:3));
                            end
                            %out = normrnd(mean,std,dims);
                        else
                            APP_LOG('last_error','Erroneous parameters on gaussian filler. Use [mean,std]');
                        end
                    otherwise
                        APP_LOG('last_error','Not supported filler %s',params{1});
                end
            else
                APP_LOG('last_error','Erroneous filler attributes. Use {''type'',[params]');
            end
            out = single(permute(out,[2 1 3 4])); %CAFFE EXPECTS SINGLE TYPE with dims set as WIDTHxHEIGHTxDEPTHxNUM.
        end
    end
    
end

