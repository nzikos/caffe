classdef MONITOR_WEIGHTS_GRADS_STATS < handle
    %MONITOR_WEIGHTS_GRADS_STATS This class is used to monitor:
    %   1. Weights
    %   2. Bias
    %   3. Gradients
    %
    %   Currently supports
    %   1. mean std monitor
    %   2. min max values monitor
    %   3. both (same plot)
    %
    %   THIS CLASS IS PART OF network CLASS
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   19/11/2015
    %       FOR:    VISION TEAM - AUTH
    
    properties
        name;
        which_parameters;
        data_diff;
        
        monitor_mean_std = false;
        monitor_min_max   = false;
        
        means;
        pstds;
        nstds;
        minimum;
        maximum;
        steps;

        fig;
        
        means_lines;
        
        pstds_lines;
        nstds_lines;
        
        min_lines;
        max_lines;
    end
    
    methods
        function obj = MONITOR_WEIGHTS_GRADS_STATS(params,name,which_ones,type)
            name = lower(name);
            obj.name = name;
            switch obj.name
                case 'parameters'
                    obj.data_diff = 'data';
                case 'gradients'
                    obj.data_diff = 'diff';
            end
            for i=1:length(type)
                type{i}=lower(type{i});
                switch type{i}
                    case 'mean_std'
                        obj.monitor_mean_std = true;    
                    case 'min_max'
                        obj.monitor_min_max  = true;
                    otherwise
                        APP_LOG('last_error','Unknown monitor type');
                end
            end
            for i=1:length(which_ones)
                which_ones{i}=lower(which_ones{i});
                switch which_ones{i}
                    case 'weights'
                        obj.which_parameters(1)=true;
                    case 'bias'
                        obj.which_parameters(2)=true;
                    otherwise
                        APP_LOG('last_error','Unknown parameter type');
                end
            end
            plot_name{1}   = ['Weights ' obj.name];
            plot_name{2}   = ['Bias ' obj.name];
            obj.steps = 1;            
            obj.fig = figure;
            for j=1:length(which_ones)
                if obj.which_parameters(j)
                    APP_LOG('info','Loading %s stats plot',plot_name{j});
                    obj.means{j}     = zeros(length(params),1);
                    %('name',[plot_name{j} ' statistics']);

                    set(obj.fig,'Colormap','default');
                    colorsMap        = obj.fig.Colormap;
                    subplot(length(which_ones),1,j)
                    if obj.monitor_mean_std
                        for i=1:length(params)
                            try
	                            obj.means{j}(i)      = gather(mean(mean(mean(mean(params(i).(obj.data_diff){j})))));
	                            obj.means_lines{j,i} = animatedline('color',colorsMap(1+(i-1)*floor(length(colorsMap)/length(params)),:));
	                            addpoints(obj.means_lines{j,i},obj.steps,obj.means{j}(i));                                
                            catch
	                            obj.means{j}(i)      = double(0);
	                            obj.means_lines{j,i} = animatedline();
	                            addpoints(obj.means_lines{j,i},obj.steps,obj.means{j}(i));                                
                            end
                        end
                        for i=1:length(params)
                            try
	                            tmp = gather(std(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j}))));
	                            obj.pstds{j}(i) = double(obj.means{j}(i) + tmp);
	                            obj.nstds{j}(i) = double(obj.means{j}(i) - tmp);
	                            obj.pstds_lines{j,i}  = animatedline('color',colorsMap(1+(i-1)*floor(length(colorsMap)/length(params)),:),'LineStyle','--');
	                            obj.nstds_lines{j,i}  = animatedline('color',colorsMap(1+(i-1)*floor(length(colorsMap)/length(params)),:),'LineStyle','--');
	                            addpoints(obj.pstds_lines{j,i},obj.steps,obj.pstds{j}(i));
	                            addpoints(obj.nstds_lines{j,i},obj.steps,obj.nstds{j}(i));
                            catch
	                            obj.pstds{j}(i) = double(0);
	                            obj.nstds{j}(i) = double(0);
	                            obj.pstds_lines{j,i}  = animatedline();
	                            obj.nstds_lines{j,i}  = animatedline();
	                            addpoints(obj.pstds_lines{j,i},obj.steps,obj.pstds{j}(i));
	                            addpoints(obj.nstds_lines{j,i},obj.steps,obj.nstds{j}(i));                            
                            end
                        end
                    end
                    
                    if obj.monitor_min_max
                        for i=1:length(params)
                            try
	                            obj.minimum{j}(i)   = double(gather(min(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j})))));
	                            obj.maximum{j}(i)   = double(gather(max(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j})))));
	                            obj.min_lines{j,i}  = animatedline('color',colorsMap(1+(i-1)*floor(length(colorsMap)/length(params)),:),'LineStyle','-.');
	                            obj.max_lines{j,i}  = animatedline('color',colorsMap(1+(i-1)*floor(length(colorsMap)/length(params)),:),'LineStyle','-.');
	                            addpoints(obj.min_lines{j,i},obj.steps,obj.minimum{j}(i));
	                            addpoints(obj.max_lines{j,i},obj.steps,obj.maximum{j}(i));
                            catch
	                            obj.minimum{j}(i)   = double(0);
	                            obj.maximum{j}(i)   = double(0);
	                            obj.min_lines{j,i}  = animatedline();
	                            obj.max_lines{j,i}  = animatedline();
	                            addpoints(obj.min_lines{j,i},obj.steps,obj.minimum{j}(i));
	                            addpoints(obj.max_lines{j,i},obj.steps,obj.maximum{j}(i));                            
                            end
                        end
                    end
                    title([plot_name{j} ' statistics']);
                    xlabel('Iterations');
                end
            end
            for i=1:length(params)
                names{i} = params(i).name;
            end
            if (obj.which_parameters(1))
                if obj.monitor_mean_std
                    legend([obj.means_lines{1,:}],names,'location','southoutside','Orientation','horizontal');
                else
                    legend([obj.min_lines{1,:}],names,'location','southoutside','Orientation','horizontal');
                end
            else
                if obj.monitor_mean_std
                    legend([obj.means_lines{2,:}],names,'location','southoutside','Orientation','horizontal');
                else
                    legend([obj.min_lines{2,:}],names,'location','southoutside','Orientation','horizontal');
                end
            end
        end
        
        function update(obj,params)
            obj.steps=obj.steps+1;
            for j=1:length(obj.which_parameters)
                for i=1:length(params)
                    if obj.which_parameters(j) %weights/bias
                        try
                            if obj.monitor_mean_std
                                obj.means{j}(i)=gather(mean(mean(mean(mean(params(i).(obj.data_diff){j})))));
                                addpoints(obj.means_lines{j,i},obj.steps,obj.means{j}(i));
                                tmp = gather(std(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j}))));
                                obj.pstds{j}(i) = double(obj.means{j}(i) + tmp);
                                obj.nstds{j}(i) = double(obj.means{j}(i) - tmp);
                                addpoints(obj.pstds_lines{j,i},obj.steps,obj.pstds{j}(i));
                                addpoints(obj.nstds_lines{j,i},obj.steps,obj.nstds{j}(i));
                            end
                            if obj.monitor_min_max
                                obj.minimum{j}(i)   = double(gather(min(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j})))));
                                obj.maximum{j}(i)   = double(gather(max(reshape(params(i).(obj.data_diff){j},1,numel(params(i).(obj.data_diff){j})))));
                                addpoints(obj.min_lines{j,i},obj.steps,obj.minimum{j}(i));
                                addpoints(obj.max_lines{j,i},obj.steps,obj.maximum{j}(i));
                            end
                        catch
                        end
                    end
                end
            end
            drawnow limitrate;
        end
    end
    
end

