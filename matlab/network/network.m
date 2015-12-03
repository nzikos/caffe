classdef network < handle
%NETWORK front-end of CNN training toolbox.
%        Used to initialize and run the training/validation procedure.
%        All submodels are part of this model.
%
%% SUB-MODELS
%        CAFFE
%        BATCH_FACTORY
%        TRAIN
%        LR_POLICY
%        MONITOR_TRAIN_ERROR
%        VALIDATION
%        MONITOR_VALIDATION
%        EXIT_TRAIN
%
%% SETTERS
%
%           set_validation_interval : Sets number of training iterations
%                                     between successive validations.
%
%           set_max_iterations      : Sets maximun number of iterations in
%                                     order to stop training session.
%
%           set_snapshot_time       : Sets number of minutes between
%                                     successive network snapshots.
%
%           set_display             : Sets number of training iterations in
%                                     order to update console
%
%           monitor                 : Sets monitoring type of
%                                     weights/bias and their respective gradients
%           
%% METHODS
%           start                   : Initiates a training session.
%
%           print_state             : Prints the state.
%
%           take_snapshot           : Takes a session's state snapshot.
%
%           tic_toc_snapshot        : sub-function of take_snapshot.
%
%           load_snapshot           : Restores saved state.
%
%% AUTHOR: PROVOS ALEXIS
%  DATE:   20/5/2015
%  FOR:    VISION TEAM - AUTH

    properties
        caffe;
        batch_factory;
        structure;
        
        train                      = [];
        lr_policy                  = [];
        monitor_train_error        = [];
        
        monitor_parameters         = struct('value',false,'which_ones',[],'args',[]);
        monitor_parameters_stats   = [];
        monitor_gradients          = struct('value',false,'which_ones',[],'args',[]);
        monitor_gradients_stats    = [];
        
        validation;
        monitor_validation         = [];
        
        iter                       = 0;
        iters_per_display 	       = 1;
        iters_per_val              = 2000;
        
        max_iterations             = inf;
        exit_train                 = EXIT_HANDLER();
        
        snapshot_path              = [];
        snapshot_time_in_minutes   = 60;
                       
        time_per_iter              = 0;
    end
    
    methods
        %% INIT
        function net = network(extraction_model,net_struct)
            if ~isa(net_struct,'NET_STRUCTURE')
                net.load_snapshot(net_struct);
                return;
                 %APP_LOG('last_error','Expected object of class "NET_STRUCTURE"');            
            end
            if ~isa(extraction_model,'extraction_model')
        		APP_LOG('last_error','Expected object of class "extraction_model"');
            end
            APP_LOG('debug','Enabling logs');
            APP_LOG('enable',fullfile(pwd,'LOGS',strcat('Logs_',datestr(now, 'DD_mm_YYYY_HH_MM_SS'),'.txt')));        
            net.structure     = net_struct;
            set               = extraction_model.sets.set;
            net.snapshot_path = fullfile(extraction_model.paths.cache,'snapshots');
            handle_dir(net.snapshot_path,'create');
            best_model_path   = fullfile(extraction_model.paths.cache,'best_model.mat');
            net.caffe         = CAFFE(net.structure);
            net.batch_factory = BATCH_FACTORY(net.structure);
            net.train         = TRAIN(net.caffe,net.batch_factory);
            net.validation    = VALIDATION(net.caffe,net.batch_factory,best_model_path);
            net.lr_policy     = LR_POLICY(net.train,net.validation,net.exit_train);
                    
            net.batch_factory.set_train_objects(extraction_model.objects.data.(set{1}));
            net.batch_factory.set_validation_objects(extraction_model.objects.data.(set{2}));
            net.batch_factory.set_mean_std(extraction_model.objects);
            net.batch_factory.set_sampling_method(true,extraction_model.get_class_frequencies('train'));
                              
            net.caffe.set_labels(extraction_model.objects.data.(set{2}));
        end
        
        %% SETTERS
        function net = set_validation_interval(net,interval)
            net.iters_per_val    = interval;
        end        
        
        function net = set_max_iterations(net,max_iters)
            net.max_iterations = max_iters;
        end
        
        function net = set_snapshot_time(net,time_in_mins)
            net.snapshot_time_in_minutes = time_in_mins;
        end
        
        function net = set_display(net,iters_per_display)
            net.iters_per_display = iters_per_display;
        end
               
        function net = monitor(net,value,name,which_ones,args)
            name = lower(name);
            switch name
                case 'parameters'
                    net.monitor_parameters.value      = value;
                    net.monitor_parameters.which_ones = which_ones;
                    net.monitor_parameters.args       = args;
                case 'gradients'
                    net.monitor_gradients.value       = value;
                    net.monitor_gradients.which_ones  = which_ones;
                    net.monitor_gradients.args        = args;                    
%		needs modification of matcaffe.cu -> caffe('get_blob_data') should return ALL feature maps.
%		Access pattern is the same so, no need to create a special case inside MATLAB
%                case 'feature_maps'
%                    net.monitor_gradients = value;
%                    if value
%                        net.monitor_gradients_stats = MONITOR_WEIGHTS_GRADS_STATS(net.caffe.get.gradients(),name,which_ones,args);
%                    end                    
                otherwise
                    APP_LOG('last_error','Unknown monitor type');
            end
        end
        
        %% TRAIN/VAL
        function net = start(net)
           
            if isempty(net.batch_factory.train_queue)
                APP_LOG('debug','Async load batch/es in host/cpu memory');                
                net.batch_factory.prepare_train_batch();
            else
                APP_LOG('debug','Async load last saved batch/es in host/cpu memory');
                for i=1:net.batch_factory.train_queue_size
                    net.batch_factory.async_load_current_train_batch(i);
                end
            end
            
            net.caffe.init('train');            
            APP_LOG('header','Training');
            if net.train.fetch_train_error
                net.monitor_train_error = MONITOR_TRAIN_ERROR(net.structure.layers{end}.type_train,net.train.error);
            end
            net.monitor_validation = MONITOR_VALIDATION(net.validation);
            if net.monitor_parameters.value
                net.monitor_parameters_stats = MONITOR_WEIGHTS_GRADS_STATS(net.structure.params,'parameters',net.monitor_parameters.which_ones,net.monitor_parameters.args);
            end
            if net.monitor_gradients.value
                net.monitor_gradients_stats = MONITOR_WEIGHTS_GRADS_STATS(net.caffe.get.gradients(),'gradients',net.monitor_gradients.which_ones,net.monitor_gradients.args);
            end
            while ~net.exit_train.read_flag()

                h=tic;
                if(net.monitor_gradients.value)
                    gradients = net.train.do_train();
                    net.monitor_gradients_stats.update(gradients);
                else
                    net.train.do_train();
                end
                if(net.train.fetch_train_error)            
                    net.monitor_train_error.update(net.train.error,net.iters_per_display);
                end
                if(net.monitor_parameters.value)
                    net.monitor_parameters_stats.update(net.structure.params);
                end
                net.iter=net.iter+1;
                net.lr_policy.check_lr(net.iter);                
                net.time_per_iter=net.time_per_iter+toc(h);                
                
                net.print_state();

                if ~mod(net.iter,net.iters_per_val)
                    APP_LOG('header','Validating');
                    net.caffe.set_phase('validation');
                    net.validation.do_validation();
                    net.monitor_validation.update();
                    APP_LOG('header','Training');
                    net.caffe.set_phase('train');
                end

                stop_train=net.exit_train.read_flag();
                stop_train=stop_train || (length(net.train.error) > net.max_iterations);
                if(stop_train)
                    net.exit_train.raise_flag();
                else
                    net.exit_train.drop_flag();
                end
                net.tic_toc_snapshot();
            end
        end
        
        %% UPDATE CONSOLE
        function print_state(net)
            if ~mod(net.iter,net.iters_per_display)
                
                mean_time_per_iter = net.time_per_iter/net.iters_per_display;

                if(net.train.fetch_train_error)
                    mean_error=mean(net.train.error(end-net.iters_per_display+1:end));                
                    APP_LOG('info','iter: %d | Mean time %6.2f ms |error: %1.8f',net.iter,mean_time_per_iter*1000,mean_error);
                else
                    APP_LOG('info','iter: %d | Mean time %6.2f ms',net.iter,mean_time_per_iter*1000);
                end
                
                net.time_per_iter = 0;                              
                
            end
        end     
        
        %% SNAPSHOT HANDLER
        % tic_toc_snapshot() is a modified version of tic_toc_print from Ross Girshick
        % AUTORIGHTS
        % -------------------------------------------------------
        % Copyright (C) 2009-2012 Ross Girshick
        %
        % This file is part of the voc-releaseX code
        % (http://people.cs.uchicago.edu/~rbg/latent/)
        % and is available under the terms of an MIT-like license
        % provided in COPYING. Please retain this notice and
        % COPYING if you use this file (or a portion of it) in
        % your project.
        % -------------------------------------------------------
        function tic_toc_snapshot(net)
            persistent th;
            if isempty(th)
                th = tic();
            end
            if toc(th) > net.snapshot_time_in_minutes*60
                net.take_snapshot(fullfile(net.snapshot_path,strcat('snapshot_',datestr(now, 'DD_mm_YYYY_HH_MM'),'.mat')));
                th = tic();
            end
        end        
      
        %% SAVE - LOAD Functions
        function take_snapshot(net,path)
            APP_LOG('info','Saving network under %s',path);
            t_net.structure                = net.structure;            
            t_net.caffe                    = net.caffe;
            t_net.batch_factory            = net.batch_factory;
            t_net.train                    = net.train;
            t_net.lr_policy                = net.lr_policy;
            t_net.validation               = net.validation;
            t_net.monitor_parameters       = net.monitor_parameters;            
            t_net.iter                     = net.iter;
            t_net.iters_per_display        = net.iters_per_display;
            t_net.iters_per_val            = net.iters_per_val;
            t_net.max_iterations           = net.max_iterations;
            t_net.exit_train               = net.exit_train;
            t_net.snapshot_path            = net.snapshot_path;
            t_net.snapshot_time_in_minutes = net.snapshot_time_in_minutes;
            save(path,'t_net','-v7');
            APP_LOG('info','Saving completed');
        end

        function net = load_snapshot(net,path)
            load(path);
            APP_LOG('info','Loading Network Structure...');
            net.structure                = t_net.structure;
            APP_LOG('debug','Gathering parameters');
            for i=1:length(net.structure.params)
                for j=1:length(net.structure.params(i).data)
                    net.structure.params(i).data{j}=gather(net.structure.params(i).data{j});
                end
            end            
            APP_LOG('info','Loading Caffe state...');
            net.caffe                    = t_net.caffe;
            APP_LOG('info','Loading Batch Factory state...');
            net.batch_factory            = t_net.batch_factory;
            APP_LOG('info','Loading training entity...');
            net.train                    = t_net.train;
            net.lr_policy                = t_net.lr_policy;            
            APP_LOG('info','Loading validation entity...');
            net.validation               = t_net.validation;
            APP_LOG('info','Loading parameters...');
            net.monitor_parameters       = t_net.monitor_parameters;
            net.iter                     = t_net.iter;
            net.iters_per_display        = t_net.iters_per_display;
            net.iters_per_val            = t_net.iters_per_val;
            net.max_iterations           = t_net.max_iterations;
            net.exit_train               = t_net.exit_train;
            net.snapshot_path            = t_net.snapshot_path;
            net.snapshot_time_in_minutes = t_net.snapshot_time_in_minutes;
            APP_LOG('info','Initializing Caffe with last saved weights...');
            net.caffe.init('train');
            APP_LOG('info','Snapshot load completed');
        end
    end 
end