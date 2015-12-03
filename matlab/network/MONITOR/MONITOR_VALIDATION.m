classdef MONITOR_VALIDATION < handle
    %MONITOR_VALIDATION This class is used to plot validation metrics.
    %
    %   THIS CLASS IS PART OF network() CLASS
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   21/11/2015
    %       FOR:    VISION TEAM - AUTH
    
    properties
        validation;
        
    	fig            = [];
        
        top1_line      = [];
        mean_top1_line = [];      
        
        topk_line      = [];
        mean_topk_line = [];
    end
    
    methods

        function obj = MONITOR_VALIDATION(validation_arg)
            obj.validation       = validation_arg;
            obj.fig              = figure('name','Validation stats');
            obj.top1_line        = animatedline('Color','r');
            obj.mean_top1_line   = animatedline('Color',[0.9882 0.5373 0.6745],'Marker','o');
            obj.topk_line        = animatedline('Color','g');
            obj.mean_topk_line   = animatedline('Color',[0.0039 0.1961 0.1255],'Marker','o');
            
            names{1}             = 'Overall TOP-1 accuracy';            
            names{2}             = 'Average TOP-1 accuracy';
            names{3} = ['Overall TOP-' num2str(obj.validation.k) ' accuracy'];            
            names{4} = ['Average TOP-' num2str(obj.validation.k) ' accuracy'];

            legend(names,'Location','southeast');
            addpoints(obj.top1_line,1:length(obj.validation.overall),[obj.validation.overall(:).top1]);
            addpoints(obj.mean_top1_line,1:length(obj.validation.average),[obj.validation.average(:).top1]);
            addpoints(obj.topk_line,1:length(obj.validation.overall),[obj.validation.overall(:).topk]);
            addpoints(obj.mean_topk_line,1:length(obj.validation.average),[obj.validation.average(:).topk]);
            drawnow;            
        end
        
        function update(obj)
            addpoints(obj.top1_line,length(obj.validation.overall),obj.validation.overall(end).top1);
            addpoints(obj.mean_top1_line,length(obj.validation.average),obj.validation.average(end).top1);            
            addpoints(obj.topk_line,length(obj.validation.overall),obj.validation.overall(end).topk);
            addpoints(obj.mean_topk_line,length(obj.validation.average),obj.validation.average(end).topk);
            drawnow;
        end
    end
    
end

