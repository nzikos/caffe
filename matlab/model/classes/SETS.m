classdef SETS < handle
    %Datasets that will be used. The extraction_model is expecting to
    %see the metadata and the images root directories contain two or more 
    %subfolders with the sets names under "1-1" relation.
    
    %   AUTHOR: PROVOS ALEXIS
    %   DATE:   19/5/2015
    %   FOR:    vision team - AUTH
    
    properties
        set={};
    end
    
    methods
        function obj = SETS(a)
            if length(a)<2
                APP_LOG('last_error','Wrong number of sets provided: %d',length(a));
            end
            obj.set=a;
        end
    end
    
end

