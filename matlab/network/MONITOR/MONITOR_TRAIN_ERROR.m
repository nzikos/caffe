classdef MONITOR_TRAIN_ERROR < handle
    %MON_TRAIN_ERROR This class is used to monitor training error when
    %asked to
    %
    %   THIS CLASS IS PART OF network CLASS
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   19/11/2015
    %       FOR:    VISION TEAM - AUTH
    properties
        fig        = []; 
        error_line = [];
        error_SMA  = [];
    end
    
    methods
        function obj = MONITOR_TRAIN_ERROR(loss_name,error_line)
                APP_LOG('info','Loading training error plot');
                obj.fig = figure('name','Training error');
                obj.error_line = animatedline('color',[110 181 254]/255);
                obj.error_SMA   = animatedline('color',[150  75   0]/255,'LineStyle','--');
                legend(loss_name,'Simple moving average');
                title([loss_name ' training error']);
                xlabel('Iteration');
                ylabel('Error');
                addpoints(obj.error_line,1:length(error_line),error_line);                            
        end
        
        function update(obj,error_line,n)
            addpoints(obj.error_line,length(error_line),error_line(end));
            if length(error_line)>n
                addpoints(obj.error_SMA,length(error_line),mean(error_line(end-n+1:end)));
            else
                addpoints(obj.error_SMA,length(error_line),mean(error_line(2:end)));
            end
            drawnow limitrate;                        
        end
    end
    
end

