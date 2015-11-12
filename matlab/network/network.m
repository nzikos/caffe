classdef network < handle
%NETWORK Summary of this class goes here
%   Detailed explanation goes here
    
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
    properties
        caffe;
        batch_factory;
        structure;
        
        train;
        training_fig        = []; 
        training_error_line = [];
        
        validation;
        validation_fig;
        validation_top1_line;
        validation_mean_top1_line;
        validation_topk_line;
        validation_mean_topk_line;
        
        iter;
        iters_per_display;
        iters_per_val;
        
        max_iterations;
        exit_train;
        
        snapshot_path;
        snapshot_time_in_minutes;
                       
        time_per_iter;        
    end
    
    methods
        %% INIT
        function net = network(extraction_model,net_struct)
            net.iter                     = 0;
            net.iters_per_val            = 2000;
            net.max_iterations           = [];
            net.snapshot_path            = [];
            net.snapshot_time_in_minutes = 60;
            net.exit_train               = EXIT_HANDLER();
            net.time_per_iter            = 0;
            if isa(net_struct,'NET_STRUCTURE')
%                 net.structure.prototxt_path    = net_struct.prototxt_path;
%                 net.structure.train_batch_size = net_struct.train_batch_size;
%                 net.structure.val_batch_size   = net_struct.val_batch_size;
%                 net.structure.test_batch_size  = net_struct.test_batch_size;
%                 net.structure.objects_size     = net_struct.objects_size;
%                 net.structure.labels_size      = net_struct.labels_size;
%                 
%                 net.structure.layers           = net_struct.layers;
%                 net.structure.params           = net_struct.params;
%                 net.structure.lr_mult          = net_struct.lr_mult;
%                 net.structure.wd_mult          = net_struct.wd_mult;
%                 
%                 net.structure.counter          = net_struct.counter;
%                 net.structure.params_counter   = net_struct.params_counter;
%                 net.structure.sub_counter      = net_struct.sub_counter;
%                 
%                 net.structure.valid_structure  = net_struct.valid_structure;
                net.structure = net_struct;
                if isa(extraction_model,'extraction_model')
                    set              = extraction_model.sets.set;

                    net.snapshot_path= fullfile(extraction_model.paths.cache,'snapshots');
                    handle_dir(net.snapshot_path,'create');
                    best_model_path= fullfile(extraction_model.paths.cache,'best_model.mat');

                    net.caffe         = CAFFE(net.structure);

                    net.batch_factory = BATCH_FACTORY(net.structure);
                    net.batch_factory.set_train_objects(extraction_model.objects.data.(set{1}));
                    net.batch_factory.set_validation_objects(extraction_model.objects.data.(set{2}));
                    net.batch_factory.set_use_mean_std(extraction_model.objects);

                    net.train        = TRAIN(net.caffe,net.batch_factory);
                    net.validation   = VALIDATION(net.caffe,net.batch_factory,best_model_path);
                    
                    net.caffe.set_labels(extraction_model.objects.data.(set{2}));                    
                else
                        APP_LOG('last_error','Expected object of class "extraction_model"');                    
                end
            else
                APP_LOG('last_error','Expected object of class "NET_STRUCTURE"');
            end
        end
        
        %% SETTERS              
        function net = set_batches_per_iter(net,batches_per_iter)
            net.train.set_batches_per_iter(batches_per_iter);
        end
        
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
        
        function net = fetch_train_error(net,value)
            net.train.fetch_train_error = value;
        end
        
        %% TRAIN/VAL
        function net = start(net)
            APP_LOG('debug','Enabling logs');
            APP_LOG('enable',fullfile(pwd,'LOGS',strcat('Logs_',datestr(now, 'DD_mm_YYYY_HH_MM_SS'),'.txt')));
                        
            if(net.train.fetch_train_error)
                APP_LOG('info','Loading training error plot');
                net.training_fig = figure('name','training error');
                net.training_error_line = animatedline('color',[110 181 254]/255);
                legend(net.structure.layers{end}.type_train);
                addpoints(net.training_error_line,1:length(net.train.error),net.train.error);                
            end
            net.init_validation_figure();            
            
            if isempty(net.batch_factory.train_queue)
                APP_LOG('debug','Async load batch/es in host/cpu memory');                
                net.batch_factory.prepare_train_batch();
            else
                APP_LOG('debug','Async load last saved batch/es in host/cpu memory');
                for i=1:net.batch_factory.train_queue_size
                    net.batch_factory.async_load_current_train_batch(i);
                end
            end
            
            APP_LOG('header','Training');
            while ~net.exit_train.read_flag()

                h=tic;
                
                net.train.do_train();
                net.update_train_error();
                net.iter=net.iter+1;

                net.time_per_iter=net.time_per_iter+toc(h);                
                
                net.print_state();

%                  x=net.caffe.get.params();
%                  [x(2).data{1}(1) x(10).data{1}(1) x(12).data{1}(1)]

                if ~mod(net.iter,net.iters_per_val)
                    APP_LOG('header','Validating');
                    net.caffe.set_phase('validation');
                    net.validation.do_validation();
                    net.update_validation_stats();
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
        %% INIT FIGURES
        function init_validation_figure(net)
            net.validation_fig              = figure('name','Validation stats');
            net.validation_top1_line        = animatedline('Color','r','Marker','*','LineStyle','-');
            net.validation_mean_top1_line   = animatedline('Color',[0.9882 0.5373 0.6745],'Marker','*','LineStyle','-');
            net.validation_topk_line        = animatedline('Color','g','Marker','*','LineStyle','-');
            net.validation_mean_topk_line   = animatedline('Color',[0.0039 0.1961 0.1255],'Marker','*','LineStyle','-');            
            legend('Overall TOP-1 accuracy','Average TOP-1 accuracy',['Overall TOP-' num2str(net.validation.k) ' accuracy'],['Average TOP-' num2str(net.validation.k) ' accuracy'],'Location','southeast');
        end
        %% UPDATE FIGURES
        function update_train_error(net)
            if(net.train.fetch_train_error)            
                addpoints(net.training_error_line,length(net.train.error),net.train.error(end));
                drawnow limitrate;            
            end
        end
        function update_validation_stats(net)
            addpoints(net.validation_top1_line,length(net.validation.overall),net.validation.overall(end).top1);
            addpoints(net.validation_mean_top1_line,length(net.validation.average),net.validation.average(end).top1);            
            addpoints(net.validation_topk_line,length(net.validation.overall),net.validation.overall(end).topk);
            addpoints(net.validation_mean_topk_line,length(net.validation.average),net.validation.average(end).topk);
            drawnow;            
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
        function tic_toc_snapshot(net)
            persistent th;
            if isempty(th)
                th = tic();
            end
            if toc(th) > net.snapshot_time_in_minutes*60
                net.save_current_network(fullfile(net.snapshot_path,strcat('snapshot_',datestr(now, 'DD_mm_YYYY_HH_MM'),'.mat')));
                th = tic();
            end
        end        
      
        %% SAVE - LOAD Functions
        function save_current_network(net,path)
            APP_LOG('info','Saving network under %s',path);
            t_net.structure                = net.structure;            
            t_net.caffe                    = net.caffe;
            t_net.batch_factory            = net.batch_factory;
            t_net.train                    = net.train;
            t_net.validation               = net.validation;
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
            APP_LOG('info','Loading Caffe state...');
            net.caffe                    = t_net.caffe;
            APP_LOG('info','Loading Batch Factory state...');
            net.batch_factory            = t_net.batch_factory;
            APP_LOG('info','Loading training entity...');
            net.train                    = t_net.train;
            APP_LOG('info','Loading validation entity...');
            net.validation               = t_net.validation;
            APP_LOG('info','Loading parameters...');
            net.iter                     = t_net.iter;
            net.iters_per_display        = t_net.iters_per_display;
            net.iters_per_val            = t_net.iters_per_val;
            net.max_iterations           = t_net.max_iterations;
            net.exit_train               = t_net.exit_train;
            net.snapshot_path            = t_net.snapshot_path;
            net.snapshot_time_in_minutes = t_net.snapshot_time_in_minutes;
            APP_LOG('info','Initializing Caffe with last saved weights...');
            net.caffe.init('train');

            APP_LOG('info','Loading validation plot...');
            addpoints(net.validation_top1_line,2:length(net.validation.overall),[net.validation.overall(2:end).top1]);
            addpoints(net.validation_mean_top1_line,2:length(net.validation.average),[net.validation.average(2:end).top1]);
            addpoints(net.validation_topk_line,2:length(net.validation.overall),[net.validation.overall(2:end).topk]);
            addpoints(net.validation_mean_topk_line,2:length(net.validation.average),[net.validation.average(2:end).topk]);
            APP_LOG('info','Snapshot load completed');
        end
    end 
end

