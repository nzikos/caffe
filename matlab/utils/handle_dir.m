%% --SUB FUNCTION WHICH HANDLES DIRECTORIES
% Depending to behavior handle_dir creates the specified directory or
% throws error.
function handle_dir(directory,behavior)
    switch behavior
        case 'create'
            if ~exist(directory,'dir')
                res=mkdir(directory);
                if(res)
                    APP_LOG('info',4,'Created directory %s',directory);
                else
                    error('Permission denied while creating %s',directory);                    
                end
            else
                APP_LOG('info',4,'Directory %s exists',directory);                
            end
        case 'throw error'
            if ~exist(directory,'dir')
                error('Directory %s does not exist',directory);
            else
                APP_LOG('info',4,'Directory %s exists',directory);                
            end
        otherwise
            error('No such behavior: "%s" for hndl_dir',behavior);
    end
end