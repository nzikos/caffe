classdef deployed_model3 < handle
    properties
        structure;
        caffe;
        batch_factory;
    end
    methods
        function net = deployed_model3(model_path,use_gpu)
            APP_LOG('info','Loading model');
            load(model_path);           
            
            APP_LOG('debug','Initializing sub-models');
            net.structure               = model.structure;
            net.structure.prototxt_path = pwd;
            net.caffe                   = CAFFE(net.structure);
            net.caffe.labels            = model.labels;
            net.caffe.set_use_gpu(use_gpu);
            
            net.batch_factory           = BATCH_FACTORY(net.structure);
            net.batch_factory.set_norm_type(model.normalization_type);
            net.batch_factory.set_mean_std(model);
            net.batch_factory.set_epsilon(model.epsilon);
                        
            APP_LOG('debug','Gathering parameters');
            for i=1:length(net.structure.params)
                for j=1:length(net.structure.params(i).data)
                    net.structure.params(i).data{j}=gather(net.structure.params(i).data{j});
                end
            end
            
            APP_LOG('debug','Setting up CAFFE');
            net.caffe.reset_object_input('test',2);            
            net.caffe.init('test');
        end
        
        function [class,conf] = predict(net,img)
            net_input = [net.structure.objects_size net.structure.test_batch_size]; %HxWxDxN
            
            t_batch = uint8(zeros(net_input));          
            img = imresize(img,net_input(1:2),'bilinear', 'antialiasing', false); %yet again, SPP
            t_batch(:,:,:,1)=img(1:net_input(1),1:net_input(2),:);
            t_batch(:,:,:,2)=flip(t_batch(:,:,:,1),2);
            batch = net.batch_factory.create_test_batch(t_batch);
            net.caffe.set.input(batch);
            net.caffe.action.forward();
            t_preds = net.caffe.get.output();
            t_preds_1 = mean(t_preds{1},4);
            preds   = squeeze(t_preds_1);
            [conf,pos] = sort(preds,'descend');
            for i=1:length(conf)
                class{i} = net.caffe.labels(pos(i)).name;
            end
        end
	end
end
        