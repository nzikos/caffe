function APP_LOG(type,fmt,varargin)
%APP_LOG Summary This function is used to initialize log keeping and
%control the console output of the toolbox.
%
%% TODO:PARAMETRIZE DEBUG option
%
%% AUTHOR: PROVOS ALEXIS
%  DATE:   20/5/2015
%  FOR:    VISION TEAM - AUTH
debug=1;
persistent log_dir;

%2. Build message    
time=sprintf('[%s] ',datestr(now, 'DD-mm-YYYY HH:MM:SS'));    
switch type
    case 'header'
        out=upper([time sprintf(fmt,varargin{:})]);
    case 'warning'
        out=[time 'WARNING: ' sprintf(fmt,varargin{:})];
    case 'enable'
        log_dir = fmt;
        [logs_dir,~,~]  =   fileparts(log_dir);
        if ~exist(logs_dir,'dir')
            try
                mkdir(logs_dir);
            catch err
                error(err.message);
            end
        end
        APP_LOG('info','LOGS kept under %s',log_dir);
        return;        
    otherwise
        out=[time sprintf(fmt,varargin{:})];
end
%3. Print message on matlab console
switch type
    case 'header'
        fprintf('\n%s\n%s\n',out,repmat('-',1,length(out)));
    case 'info'
        fprintf('%s\n',out);
    case 'debug'
        if debug
            fprintf('%s\n',out);
        end
    case 'error'
        fprintf(2,'%s\n',out);
    case 'warning'
        sound(audioread('/utils/Beep-SoundBible.com-923660219.wav'));        
        fprintf(2,'%s\n',out);
    case 'last_error'
        sound(audioread('/utils/Beep-SoundBible.com-923660219.wav'));
        error(out);
    otherwise
        error('Wrong type of msg passed to APP_LOG');
end
%4. Keep message on logs
if ~isempty(log_dir)
    fileID=fopen(log_dir,'a');
    fprintf(fileID,'%s\n',out);
    fclose(fileID);
end
end
