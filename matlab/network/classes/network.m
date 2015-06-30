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
        train;
        validation;

        iter;
        epoch;

        iters_per_epoch;
        iters_per_val;
        
        max_iterations;
        val_target;
        
        snapshot_path;
        snapshot_time_in_minutes;
    end
    
    methods
        %% INIT
        function net = network()
            net.caffe                    = CAFFE();
            net.train                    =[];
            net.validation               =[];
            net.iter                     =0;
            net.epoch                    =0;
            net.iters_per_epoch          =[];
            net.iters_per_val            =[];   
            net.max_iterations           =[];
            net.val_target               =[];      
            net.snapshot_path            =[];
            net.snapshot_time_in_minutes =60;
        end
        
        %% SETTERS
        function net = set_model(net,extraction_model)
            if strcmp(class(extraction_model),'extraction_model')
                            
                set              = extraction_model.sets.set;
                
                net.snapshot_path= fullfile(extraction_model.paths.cache,'snapshots');
                handle_dir(net.snapshot_path,'create');
                best_weights_path= fullfile(extraction_model.paths.cache,'best_weights.mat');
                
                net.train        = TRAIN(net.caffe,extraction_model.objects.data.(set{1}));
                net.validation   = VALIDATION(net.caffe,extraction_model.objects.data.(set{2}),best_weights_path);
                
                net.caffe.set_labels(extraction_model.objects.data.(set{2}));
            else
                APP_LOG('error_last',0,'Expected class "extraction_model"');
            end
        end
              
        function net = set_batches_per_iter(net,batches_per_iter)
            net.train.set_batches_per_iter(batches_per_iter);
            net.iters_per_epoch  = floor(length(net.train.objects)/(batches_per_iter*(net.caffe.batch_factory.batch_size/(1+net.caffe.batch_factory.flip))));
        end
        
        function net = set_validations_per_epoch(net,vals_per_epoch)
            net.iters_per_val    = round(net.iters_per_epoch/vals_per_epoch);
        end
        
        function net = set_max_iterations(net,max_iters)
            net.max_iterations = max_iters;
        end
        
        function net = set_target(net,target_error)
            net.val_target = target_error;
        end                
        
        function net = set_snapshot_time(net,time_in_mins)
            net.snapshot_time_in_minutes = time_in_mins;
        end
        %% TRAIN/VAL
        function net = start(net)
            APP_LOG('header',0,'Training');
            net.caffe.set.phase.train();
            while (length(net.train.error) < net.max_iterations)&&(net.validation.error(end) > net.val_target)
                net.train.do_train();
                net.iter=net.iter+1;

                APP_LOG('info',0,'iter: %d/%d | epoch %d/%d | error: %1.15f',net.iter,net.iters_per_epoch,net.epoch,floor(net.max_iterations/net.iters_per_epoch),net.train.error(end));

                if ~mod(net.iter,net.iters_per_val) || net.iter==net.iters_per_epoch
                    APP_LOG('header',0,'Validating');
                    net.caffe.set.phase.test();
                    net.validation.do_validation();                    
                    APP_LOG('header',0,'Training');
                    net.caffe.set.phase.train();
                end
                
                if net.iter>=net.iters_per_epoch
                    net.iter=0;
                    net.epoch=net.epoch+1;
                end

                net.tic_toc_snapshot();
            end
        end
        %% SNAPSHOT HANDLERS
        function tic_toc_snapshot(net)
            persistent th;
            if isempty(th)
                th = tic();
            end
            if toc(th) > net.snapshot_time_in_minutes*60
                net.force_snapshot('tic_toc_snapshot.mat');
                th = tic();
            end
        end
        
        function force_snapshot(net,name)
            fullpath = fullfile(net.snapshot_path,name);
            APP_LOG('info',0,'Taking network snapshot under %s',fullpath);
            save(fullpath,'net') %<<--- add 'net,'-v6') to skip compression phase
            APP_LOG('info',0,'Snapshot taken');
        end

        function net = load_snapshot(net,path)
            tmp = load(path);
            net.caffe                    = tmp.net.caffe;
            net.train                    = tmp.net.train;
            net.validation               = tmp.net.validation;
            net.iter                     = tmp.net.iter;
            net.epoch                    = tmp.net.epoch;
            net.iters_per_epoch          = tmp.net.iters_per_epoch;
            net.iters_per_val            = tmp.net.iters_per_val;
            net.max_iterations           = tmp.net.max_iterations;
            net.val_target               = tmp.net.val_target;
            net.snapshot_path            = tmp.net.snapshot_path;
            net.snapshot_time_in_minutes = tmp.net.snapshot_time_in_minutes;
            net.caffe.set.weights();
        end
    end
end

