classdef VALIDATION < handle
%VALIDATION Class used to perform validation and store validation results
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH
    properties
        caffe;
        batch_factory;
        
        k;        
        best_model_path;
        per_class_stats;  
        average        = struct('error',[],'top1',[],'topk',[]);
        overall        = struct('error',[],'top1',[],'topk',[]);
        best           = struct('error',[],'top1',[],'topk',[]);
        found_new_best = 0;
        target_type    = 'average';
        target_param   = 'top1';
    end
    
    methods
        %% INIT
        function val = VALIDATION(arg_caffe,arg_batch_factory,arg_best_model_path)
            val.caffe           = arg_caffe;
            val.batch_factory   = arg_batch_factory;
            
            val.overall.error   = inf;
            val.overall.top1    = 0;            
            val.overall.topk    = 0;            
            val.average.error   = inf;            
            val.average.top1    = 0;
            val.average.topk    = 0;
            
            val.best_model_path = arg_best_model_path;
            val.best.error      = inf;
            val.best.top1       = 0;
            val.best.topk       = 0;
        end
        %% SETTERS
        function set_k(val,arg_k)
            val.k=arg_k;
        end
        function set_best_target(val,type,param)
            type = lower(type);
            param= lower(param);
            switch type
                case 'overall'
                    val.target_type = type;
                case 'average'
                    val.target_type = type;
                otherwise
                    APP_LOG('last_error','Invalid validation type. Set Overall/Average');
            end
            switch param
                case 'error'
                    val.target_param = param;
                case 'top1'
                    val.target_param = param;
                case 'topk'
                    val.target_param = param;
                otherwise
                    APP_LOG('last_error','Invalid validation param. Set Error/Top1/Topk');
            end
        end
        %% FUNCTIONS 
        function do_validation(val)
                rcaffe   = val.caffe;
                bfactory = val.batch_factory;
                
                val.per_class_stats         = zeros(length(rcaffe.labels),4);

                batch = bfactory.prepare_validation_batch();

                while(~isempty(batch))
                    rcaffe.set.input(batch(1));                    
                    rcaffe.action.forward();
                    probs=rcaffe.get.output();
                    
                    val.sum_per_class_stats(batch(2),probs{1});
                    batch = bfactory.create_validation_batch();                    
                end
                
                sum_stats =sum(val.per_class_stats);
                val.overall(end+1).error=sum_stats(1)/sum_stats(4);                    
                val.overall(end).top1 =sum_stats(2)/sum_stats(4);
                val.overall(end).topk =sum_stats(3)/sum_stats(4);

                val.compute_per_class_stats();
                val.print_per_class_stats();
                val.average(end+1).error = mean(val.per_class_stats(:,1));
                val.average(end).top1  = mean(val.per_class_stats(:,2));
                val.average(end).topk  = mean(val.per_class_stats(:,3));

                APP_LOG('info','Overall results: Error: %1.4f Top-1: %5.2f%% Top-%d: %5.2f%%',val.overall(end).error,val.overall(end).top1*100,val.k,val.overall(end).topk*100);
                APP_LOG('info','Average results: Error: %1.4f Top-1: %5.2f%% Top-%d: %5.2f%%',val.average(end).error,val.average(end).top1*100,val.k,val.average(end).topk*100);
                APP_LOG('info','Best    results: Error: %1.4f Top-1: %5.2f%% Top-%d: %5.2f%%',val.best.error,val.best.top1*100,val.k,val.best.topk*100);
                APP_LOG('info','Best    Type:  %s     | param: %s',upper(val.target_type),upper(val.target_param));
                
                % Save IF best model
                %if val.error(end) < min(val.error(1:end-1))
                switch val.target_param
                    case 'error'
                        if val.(val.target_type)(end).(val.target_param) < val.best.error
                            val.save_best_model();
                            val.best.error       = val.(val.target_type)(end).error;
                            val.best.top1        = val.(val.target_type)(end).top1;
                            val.best.topk        = val.(val.target_type)(end).topk;
                            val.found_new_best   = 1;
                        else
                            val.found_new_best   = 0;
                        end
                    otherwise
                        if val.(val.target_type)(end).(val.target_param) > val.best.(val.target_param)
                            val.save_best_model();
                            val.best.error       = val.(val.target_type)(end).error;
                            val.best.top1        = val.(val.target_type)(end).top1;
                            val.best.topk        = val.(val.target_type)(end).topk;
                            val.found_new_best   = 1;
                        else
                            val.found_new_best   = 0;
                        end                        
                end
        end            
        
        function save_best_model(val)
                    model.params             = val.caffe.get.params();
                    model.structure          = val.caffe.structure;
                    model.labels             = val.caffe.labels;
                    model.per_class_stats    = val.per_class_stats;
                    model.average            = val.average;
                    model.overall            = val.overall;
                    model.normalization_type = val.batch_factory.normalization_type;                    
                    model.mean               = val.batch_factory.mean;
                    model.std                = val.batch_factory.std;
                    save(val.best_model_path,'model','-v6');
                    clear model;                    
                    APP_LOG('info','Best model saved\n');
        end
        
        function sum_per_class_stats(val,batch,results)
            [~,sorted_prediction_ids]=sort(results,3,'descend');
            sorted_prediction_ids=squeeze(sorted_prediction_ids)';
            uids = squeeze(batch{1}) + 1; %caffe is 0-based
            
            %Multinomial logistic Loss
            %-------------------------
            %results(results<realmin('single'))=realmin('single');
            results(results<eps)=eps;
            for i=size(results,4):-1:1
                x(i) = - log(results(1,1,uids(i),i));
            end
            y=unique(uids);
            for i=1:length(y)
                val.per_class_stats(y(i),1)=val.per_class_stats(y(i),1)+sum(x(y(i)==uids));
            end
            %-------------------------
            %Extract mse per class
%              x=sum((batch{2}-results).^2,3);
%              x=squeeze(x);
%              y=unique(uids);
%              for i=1:length(y)
%                  val.per_class_stats(y(i),1)=val.per_class_stats(y(i),1)+sum(x(y(i)==uids));
%              end
            
            %Extract Accuracy - TOP1
            acc_indices=uids.*(sorted_prediction_ids(:,1) == uids);
            acc_indices=acc_indices(acc_indices~=0); %remove inaccurate res
            if ~isempty(acc_indices)
                tbl = tabulate(acc_indices);
                val.per_class_stats(tbl(:,1),2)=val.per_class_stats(tbl(:,1),2)+tbl(:,2);
            end

            %Extract Accuracy - TOPK
            rep_uids=repmat(uids,1,val.k);       
            acc_indices=uids.*(sum(sorted_prediction_ids(:,1:val.k)==rep_uids,2));
            acc_indices=acc_indices(acc_indices~=0); %remove inaccurate res
            if ~isempty(acc_indices)
                tbl = tabulate(acc_indices);
                val.per_class_stats(tbl(:,1),3)=val.per_class_stats(tbl(:,1),3)+tbl(:,2);
            end

            %Add processed objects on stats tables
            tbl = tabulate(uids);
            val.per_class_stats(tbl(:,1),4)=val.per_class_stats(tbl(:,1),4)+tbl(:,2);
        end
        
        function compute_per_class_stats(val)
            val.per_class_stats(:,1)=val.per_class_stats(:,1)./val.per_class_stats(:,4);
            val.per_class_stats(:,2)=val.per_class_stats(:,2)./val.per_class_stats(:,4);
            val.per_class_stats(:,3)=val.per_class_stats(:,3)./val.per_class_stats(:,4);
        end
        
        function print_per_class_stats(val)
            APP_LOG('header','%25s %10s %7s %6s%d %7s','CLASS','error','top1','top',val.k,'Objects');
            [~,idces]=sort(val.per_class_stats(:,2));
            for i=1:length(val.per_class_stats)
                APP_LOG('info','%25s %10.4f %6.2f%% %6.2f%% %7d',val.caffe.labels(idces(i)).name,val.per_class_stats(idces(i),1),val.per_class_stats(idces(i),2)*100,val.per_class_stats(idces(i),3)*100,val.per_class_stats(idces(i),4));
            end
        end
       
        end
end

