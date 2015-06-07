classdef VALIDATION < handle
    %VALIDATION Class used to store validation parameters such as errors,
    %target errors, times validation performed
    
    properties
        caffe;
        objects;
        error;
        accuracy;
        
        val_fig = figure('name','Validation stats');
    end
    
    methods
        %% INIT
        function val = VALIDATION(arg_caffe,arg_objects)
            val.caffe       = arg_caffe;
            val.objects     = vectorize_objects_fpaths(arg_objects);
            val.error(1)    = 1;
            val.accuracy(1) = 0;
        end
        
        %% FUNCTIONS 
        function do_validation(val)
                pos=1;
                curr_accuracy=0;
                curr_mse=0;
                loops=0;
                while(length(val.objects)-pos>val.caffe.batch_size)
                    
                    batch = create_simple_batch(val.objects(pos:end),val.caffe.batch_size);

                    val.caffe.set.input(batch(1:2));
                    val.caffe.action.forward();
                    result=val.caffe.get.output();
                    
                    curr_accuracy = curr_accuracy + result{1};
                    this_mse      = mean(sum((batch{3}-result{2}).^2,3),4);
                    curr_mse      = curr_mse + this_mse;
                    loops         = loops+1;
                    pos           = pos+val.caffe.batch_size;
APP_LOG('info',0,'validation error: %1.15f accuracy: %f%%',curr_mse/loops,(curr_accuracy/loops)*100);
                end
                val.handle_output(curr_mse/loops,curr_accuracy/loops);
        end            
            
        function handle_output(val,curr_mse,curr_accuracy)
            val.error(end+1)        = curr_mse;
            val.accuracy(end+1)     = curr_accuracy;

            APP_LOG('info',0,'validation error: %1.15f accuracy: %f%%',val.error(end),val.accuracy(end)*100);
            figure(val.val_fig);
            plot(1:length(val.error),val.error,':bo',1:length(val.error),val.accuracy,':r*');
        end

        end
end

