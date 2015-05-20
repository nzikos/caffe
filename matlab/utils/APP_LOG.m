function APP_LOG(type,msg_level,fmt,varargin)
%APP_LOG Summary of this function goes here
%   Detailed explanation goes here
print_lvl=3; %values [0-4] 4 is maximum output

if msg_level<=print_lvl
%1. Build message    
    time=sprintf('[%s] ',datestr(now, 'DD-mm-YYYY HH:MM:SS'));    
    switch type
        case 'header'
            out=upper([time sprintf(fmt,varargin{:})]);
        case 'warning'
            out=[time 'WARNING: ' sprintf(fmt,varargin{:})];
        otherwise
            out=[time sprintf(fmt,varargin{:})];            
    end
%2. Print message on matlab console and logs
    diary on;
    switch type
        case 'header'
            fprintf('\n%s\n%s\n',out,repmat('-',1,length(out)));
        case 'info'
            fprintf('%s\n',out);
        case 'error'
            fprintf(2,'%s\n',out);            
        case 'warning'
            fprintf(2,'%s\n',out);
        case 'last_error'
            error(out);
        otherwise
            error('Wrong type of msg passed to APP_LOG');
    end
    diary off;
end
