classdef LR_POLICY < handle
    %LR_POLICY is used to alter the learning rate according to specific
    %gamma value which is set upon network setup
    %
    % This class is part of NETWORK class
    %
    % Supported types inherited from CAFFE
    %   1. fixed:     Never change base_lr.
    %   2. step:      Change base_lr after a certain amount of steps.
    % More types
    %   1. heuristic: checks for consecutive better models extracted from
    %                 validation set and alters the learning rate if 
    %                 it can not find a better model with current base_lr.
    %
    %% AUTHOR: PROVOS ALEXIS
    %   DATE:   19/11/2015
    %   FOR:    VISION TEAM - AUTH
    properties
        train                       = [];
        validation                  = [];
        exit_handler                = [];

        type                        = [];
        gamma                       = [];
        
        step                        = [];
        
        failed_vals                 = 0;
        failed_vals_threshold       = 0;
        failed_lr_changes           = 0;
        failed_lr_changes_threshold = 0;
        curr_val_idx                = 1;
    end
    
    methods
        function obj = LR_POLICY(train,validation,exit_handler)
            obj.type         = 'fixed';
            obj.train        = train;
            obj.validation   = validation;
            obj.exit_handler = exit_handler;
        end
        function set(obj,type,args)
            type = lower(type);
            obj.gamma                              = args{1};
            switch type
                case 'fixed'
                    obj.type                       = type;
                case 'step'
                    obj.type                       = type;
                    if ~length(args)==2
                        APP_LOG('last_error','Step lr policy error: Requires 2 arguments {gamma,step}');
                    end
                    if ~(isnumeric(args{1}) && isnumeric(args{2}))
                        APP_LOG('last_error','Step lr policy error: Requires 2 numerical {gamma,step}');
                    end
                    if ~(args{1}>0 && args{2}>0)
                        APP_LOG('last_error','Step lr policy error: Use positive values');
                    end
                    obj.step                        = args{2};
                case 'heuristic'
                    obj.type                        = type;
                    if ~length(args)==3
                        APP_LOG('last_error','Heuristic lr policy error: Requires 3 arguments {gamma,vals_threshold,lr_changes_threshold}');
                    end
                    if ~(isnumeric(args{1}) && isnumeric(args{2}) && isnumeric(args{3}))
                        APP_LOG('last_error','Heuristic lr policy error: Requires 3 numerical {gamma,vals_threshold,lr_changes_threshold}');
                    end
                    if ~(args{1}>0 && args{2}>0 && args{3}>0)
                        APP_LOG('last_error','Step lr policy error: Use positive values');
                    end
                    obj.failed_vals_threshold       = args{2};
                    obj.failed_lr_changes_threshold = args{3};
                otherwise
                    APP_LOG('last_error','Unsupported lr policy');
            end
            APP_LOG('info','lr policy set to %s',obj.type);
        end
        function check_lr(obj,iter)
            switch obj.type
                case 'fixed'
	                return;
                case 'step'
                    if ~mod(iter,obj.step)
                        APP_LOG('info','lr policy changes base learning rate from %f to %f',obj.train.method.base_lr,obj.train.method.base_lr*obj.gamma);
        %               APP_LOG('info','SGD changes momentum rate from %f to %f',obj.m,obj.m*obj.gamma + 1 - obj.gamma);
                        obj.train.method.base_lr=obj.train.method.base_lr*obj.gamma;
        %               obj.m =obj.m*obj.gamma + 1 - obj.gamma;
                    end                    
                case 'heuristic'
                    if obj.curr_val_idx < length(obj.validation.average)
                        obj.curr_val_idx=obj.curr_val_idx+1;
                        if ~obj.validation.found_new_best
                            obj.failed_vals=obj.failed_vals+1;
                            APP_LOG('info','Failed validations %d/%d',obj.failed_vals,obj.failed_vals_threshold);
                            if(obj.failed_vals>=obj.failed_vals_threshold)
                                APP_LOG('info','lr policy changes base learning rate from %f to %f',obj.train.method.base_lr,obj.train.method.base_lr*obj.gamma);
                                obj.train.method.base_lr=obj.train.method.base_lr*obj.gamma;
                                obj.failed_vals=0;
                                obj.failed_lr_changes=obj.failed_lr_changes+1;
                                if(obj.failed_lr_changes>obj.failed_lr_changes_threshold)
                                    APP_LOG('info','lr policy raised the exit flag');
                                    obj.exit_handler.raise_flag();
                                end
                            end
                        else
                            obj.failed_vals=0;
                            obj.failed_lr_changes=0;
                        end
                    end
                otherwise
                    APP_LOG('last_error','Unsupported lr policy');
            end
        end
    end
    
end
