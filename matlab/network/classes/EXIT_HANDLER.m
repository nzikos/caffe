classdef EXIT_HANDLER < handle
    %Classdef of object handling the Training stop flag
    
    properties (SetAccess = private)
        flag;
    end
    
    methods
        function obj = EXIT_HANDLER()
            obj.flag = 0;
        end
        
        function raise_flag(obj)
            obj.flag = 1;
        end
        
        function drop_flag(obj)
            obj.flag = 0;
        end
        
        function value = read_flag(obj)
            value = obj.flag;
        end
    end
    
end

