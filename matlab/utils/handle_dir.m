% handle_dir sub function which handles the behavior for specific
% directories
%
%   Options are:
%       Create: In order to create a non-existent directory
%       Throw_error: In order to throw an error if a specified directory
%       does not exist.
%
%%  AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

function handle_dir(directory,behavior)
    switch behavior
        case 'create'
            if ~exist(directory,'dir')
                res=mkdir(directory);
                if(res)
                    APP_LOG('debug','Created directory %s',directory);
                else
                    APP_LOG('last_error','Permission denied while creating %s',directory);                    
                end
            else
                APP_LOG('debug','Directory %s exists',directory);                
            end
        case 'throw error'
            if ~exist(directory,'dir')
                APP_LOG('last_error','Directory %s does not exist',directory);
            else
                APP_LOG('debug','Directory %s exists',directory);                
            end
        otherwise
            APP_LOG('last_error','No such behavior: "%s" for hndl_dir',behavior);
    end
end