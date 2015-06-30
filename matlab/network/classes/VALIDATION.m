classdef VALIDATION < handle
    %VALIDATION Class used to store validation parameters such as errors,
    %target errors, times validation performed
    
    properties
        caffe;
        objects;
        error;
        accuracy;
        best_weights_path;
        per_class_stats;
        k;
        
        val_fig = figure('name','Validation stats');
    end
    
    methods
        %% INIT
        function val = VALIDATION(arg_caffe,arg_objects,arg_best_weights_path)
            val.caffe             = arg_caffe;
            val.objects           = vectorize_objects_fpaths(arg_objects);
            val.error(1)          = 1;
            val.accuracy(1)       = 0;
            val.best_weights_path = arg_best_weights_path;
            %figure initialization
            %set(val.val_fig,'MenuBar','none');
        end
        %% SETTERS
        function set_k(val,arg_k)
            val.k=arg_k;
        end
        %% FUNCTIONS 
        function do_validation(val)
                rcaffe = val.caffe;
                
                % Validate
                pos=1;
                curr_accuracy=0;
                curr_mse=0;
                loops=0;
                val.per_class_stats         = zeros(length(rcaffe.labels),4);
                
                while(length(val.objects)-pos>=rcaffe.batch_factory.batch_size)
                    
                    batch = rcaffe.batch_factory.create_validation_batch(val.objects(pos:end));

                    rcaffe.set.input(batch(1:2));
                    rcaffe.action.forward();
                    result=rcaffe.get.output();
                    
                    curr_accuracy = curr_accuracy + result{1};
                    
                    this_mse      = mean(sum((batch{3}-result{2}).^2,3),4);
                    curr_mse      = curr_mse + this_mse;
                    
                    loops         = loops+1;
                    pos           = pos+rcaffe.batch_factory.batch_size;
                    
                    val.sum_per_class_stats(batch(2:3),result{2});
                    APP_LOG('info',0,'validation error: %1.15f accuracy: %f%%',curr_mse/loops,(curr_accuracy/loops)*100);
                end
                
                val.compute_per_class_stats();
                val.print_per_class_stats();
                
                % Save best weights
                if(curr_mse/loops < max(val.error))
                    weights = rcaffe.get.weights();
                    save(val.best_weights_path,'weights');
                    APP_LOG('info',0,'Best results weights saved\n');
                end
                
                % Update UI - historic data
                val.handle_output(curr_mse/loops,curr_accuracy/loops);
        end            
            
        function handle_output(val,curr_mse,curr_accuracy)
            val.error(end+1)        = curr_mse;
            val.accuracy(end+1)     = curr_accuracy;

%           APP_LOG('info',0,'validation error: %1.15f accuracy: %f%%',val.error(end),val.accuracy(end)*100);
            figure(val.val_fig);
            plot(1:length(val.error),val.error,':bo',1:length(val.error),val.accuracy,':r*');
        end

        function sum_per_class_stats(val,batch,results)
            for i=1:length(batch{1})
                classuid = batch{1}(i) + 1; % uid+1 because matlab is not zero-based
                
                %Error
                this_object_mse = sum((batch{2}(:,:,:,i)-results(:,:,:,i)).^2);              
                val.per_class_stats(classuid,1)=val.per_class_stats(classuid,1)+this_object_mse;
                
                %Accuracy
                [~,classes]=sort(results(:,:,:,i),'descend');
                if classes(1) == classuid % TOP-1
                    val.per_class_stats(classuid,2)=val.per_class_stats(classuid,2)+1;
                end
                if ~isempty(find(classuid==classes(1:val.k),1)) %TOP-k
                    val.per_class_stats(classuid,3)=val.per_class_stats(classuid,3)+1;
                end
                
                val.per_class_stats(classuid,4)=val.per_class_stats(classuid,4)+1;
            end
        end
        
        function compute_per_class_stats(val)
            for i=1:length(val.per_class_stats)
                val.per_class_stats(i,1)=val.per_class_stats(i,1)/val.per_class_stats(i,4);
                val.per_class_stats(i,2)=val.per_class_stats(i,2)/val.per_class_stats(i,4);
                val.per_class_stats(i,3)=val.per_class_stats(i,3)/val.per_class_stats(i,4);
            end
        end
        
        function print_per_class_stats(val)
            APP_LOG('header',0,'%25s %12s %8s %8s%d','CLASS','error','top1','top',val.k);
            [~,idces]=sort(val.per_class_stats(:,2));
            for i=1:length(val.per_class_stats)
                APP_LOG('info',0,'%25s %1.12f %1.2f%% %1.2f%%',val.caffe.labels(idces(i)).name,val.per_class_stats(idces(i),1),val.per_class_stats(idces(i),2)*100,val.per_class_stats(idces(i),3)*100);
            end
        end
        
        end
end

