function output = bool2str( input )
%BOOL2STR This function is used to convert boolean values to strings in
%order to be printed inside prototxt files.
%
%% AUTHOR: PROVOS ALEXIS
%  DATE:   08/11/2015
%  FOR:    VISION TEAM - AUTH

    mat = {'false','true'};
    output = mat{1+input};
end

