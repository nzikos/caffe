classdef deployed_model2 < handle
    properties 
        caffe;
        batch_factory;
        klsh;
        
        validation_meta;
        val_bboxes_dir;
        
        test_img_list;

        activation_threshold= [];
        stride              = [];
        segments_bounds     = [];
        total_subregions    = [];
        total_objects       = [];
        
        label_correspondences = [];
        percentage;
    end
    methods
        function net = deployed_model2(model_path)
            APP_LOG('info','Loading model');
            load(model_path);
            [model_dir,~,~]  = fileparts(model_path);
            
            APP_LOG('debug','Initializing caffe wrapper');
            net.caffe        = CAFFE(model_dir);
            
            APP_LOG('debug','Loading network structure');
            net.caffe.set_structure(model.structure);
            
            APP_LOG('debug','Initialize batch factory');
            net.batch_factory = BATCH_FACTORY(net.caffe.structure);
            net.batch_factory.normalization_type = model.normalization_type;
            net.batch_factory.mean = model.mean;
            net.batch_factory.std = model.std;
            
            APP_LOG('debug','Loading labels');
            net.caffe.labels = model.labels;
            
            APP_LOG('debug','Setting model weights and bias');
            net.caffe.params = model.params;
            
            APP_LOG('debug','Initializing KLSH');
            net.klsh=KLSH();
            
            APP_LOG('debug','Loading label_correspondences');
            tmp = load('label_correspondences.mat');
            net.label_correspondences = tmp.label_correspondences;
            
        end
        function net = load_test_imdb(net,directory,extension)
            APP_LOG('debug','Loading test images list');
            net.test_img_list = super_get_file_list(directory,extension);
            if isempty(net.test_img_list)
                APP_LOG('last_error','TEST IMAGE LIST IS EMPTY');
            end
        end
        function net = load_validation_metadata(net,val_meta)
            tmp=load(val_meta);
            net.validation_meta = tmp.data.validation;
        end
        
        function net = set_overlap(net,overlap_percentage)
            model_struct = net.caffe.structure;
            net.stride(1) = floor(model_struct.object_height * (1-overlap_percentage));
            net.stride(2) = floor(model_struct.object_width  * (1-overlap_percentage));
            APP_LOG('info','Overlap percentage set to %1.2f%%',overlap_percentage*100);
            APP_LOG('info','Model''s input size is [%d x %d]',model_struct.object_height,model_struct.object_width);
            APP_LOG('info','Stride set to [%d x %d]',net.stride(1),net.stride(2));
            if(net.stride(1)==0)
                APP_LOG('warning','Cant use a stride of 0. Changing to 1')
                net.stride(1)=1;
            end
            if(net.stride(2)==0)
                APP_LOG('warning','Cant use a stride of 0. Changing to 1')
                net.stride(2)=1;
            end            
        end
        function net = set_segments_bounds(net,segments_bounds)
            net.segments_bounds = segments_bounds;
            APP_LOG('info','Image Segmentation between [%d x %x] to [%d x %d] subregions',segments_bounds(1),segments_bounds(1),segments_bounds(2),segments_bounds(2));
            subregions = 0;
            for i=segments_bounds(1):segments_bounds(2)
                for j=segments_bounds(1):segments_bounds(2)                
                    subregions = subregions + i*j;
                end
            end
            net.total_subregions = subregions;
            APP_LOG('info','Total subregions per image: %d',subregions);
            net.total_objects    = subregions + subregions; %AND THE FLIPPED ones
            APP_LOG('info','Total objects fed into network per image: %d',net.total_objects);
        end
        
        function net = set_activation_threshold(net,arg_thres)
            net.activation_threshold = arg_thres;
        end

        function net = set_val_bboxes_dir(net,arg_dir)
            net.val_bboxes_dir = arg_dir;
        end
        
        function compute(net,dataset)
                        
            switch(dataset)
                case 'validation'
                    img_list = {net.validation_meta.imdb_cor};
                case 'test'
                    img_list = net.test_img_list;
                otherwise
                    APP_LOG('last_error','Unknown dataset. Use validation/test');
            end
            
            fileID=fopen([dataset '_results.txt'],'w+');
            
            %for i=1:length(img_list)
            for i=1:50
                APP_LOG('info','Image %d/%d',i,length(img_list));
                img = imread(img_list{i});
                if size(img,3)==1 %B&W IMAGES
                    img = repmat(img,1,1,3);
                end
              
                if(strcmp(dataset,'validation'))
                    original_size = net.validation_meta(i).size;
                    img = imresize(img,original_size);                    
                end
                
                input_size = [net.caffe.structure.object_height,net.caffe.structure.object_width];
                
                cnd_id =1;
                inner_img_counter =1;
                for segments_y=net.segments_bounds(1):net.segments_bounds(2)
                    for segments_x=net.segments_bounds(1):net.segments_bounds(2)
                        for segment_x_idx=segments_x-1:-1:0
                            for segment_y_idx=segments_y-1:-1:0
                                bndbox.ymin = uint16(ceil(segment_y_idx*net.stride(1)*original_size(1)/((segments_y-1)*net.stride(1)+input_size(1))));
                                bndbox.ymax = uint16(floor((segment_y_idx*net.stride(1)+input_size(1))*original_size(1)/((segments_y-1)*net.stride(1)+input_size(1))));
                                bndbox.xmin = uint16(ceil(segment_x_idx*net.stride(2)*original_size(2)/((segments_x-1)*net.stride(2)+input_size(2))));
                                bndbox.xmax = uint16(floor((segment_x_idx*net.stride(2)+input_size(2))*original_size(2)/((segments_x-1)*net.stride(2)+input_size(2))));
                                if bndbox.xmin==0
                                    bndbox.xmin=1;
                                end
                                if bndbox.ymin==0;
                                    bndbox.ymin=1;
                                end
                                tmp=img(bndbox.ymin:bndbox.ymax,bndbox.xmin:bndbox.xmax,:);
                                images(:,:,:,inner_img_counter) = imresize(tmp,input_size);%,'bilinear', 'antialiasing', false);
                                inner_img_counter = inner_img_counter +1;
                                images(:,:,:,inner_img_counter) = flip(images(:,:,:,inner_img_counter-1),2);
                                inner_img_counter = inner_img_counter +1;
                                
                                candidate(cnd_id).bndbox = bndbox;                                
                                cnd_id = cnd_id +1;                                
                            end
                        end
                    end
                end
                batch_size = net.caffe.structure.test_batch_size;
                if(mod(batch_size,2))
                    APP_LOG('last_error','Use multiple of 2 in test_batch_size');
                end
                
                cnd_id = 1;
                for j=1:batch_size:size(images,4)
                    thrshold = j+batch_size-1;
                    if thrshold < size(images,4)
                        batch=net.batch_factory.create_test_batch(images(:,:,:,j:thrshold));
                    else
                        batch{1}=single(zeros([size(images,1) size(images,2) size(images,3) batch_size]));
                        thrshold = size(images,4);
                        tmp=net.batch_factory.create_test_batch(images(:,:,:,j:thrshold));                        
                        batch{1}(:,:,:,1:size(images,4)-j+1) = tmp{1};
                    end
                    net.caffe.set.input(batch);
                    net.caffe.action.forward();
                    ret=net.caffe.get.output();
                    for k=1:2:size(ret{1},4)
                        if cnd_id<=length(candidate)
                            prediction = squeeze(mean(ret{1}(:,:,:,k:k+1),4));
                            candidate(cnd_id).prediction = prediction;
                            cnd_id = cnd_id +1;
                        else
                            break;
                        end
                    end
                end
                
                klsh_pred = net.klsh.get_prediction(img);
                
                for j=1:length(candidate)
                    candidate(j).prediction = candidate(j).prediction .* klsh_pred;
                    tmp_sum = sum(candidate(j).prediction);
                    candidate(j).prediction = candidate(j).prediction ./ tmp_sum;
                end
                %Find best candidates
                which_candidate=ones(length(candidate(1).prediction),1);
                max_pred =candidate(1).prediction;
                for k=2:length(candidate)
                    for j=1:length(candidate(k).prediction)
                        if max_pred(j) < candidate(k).prediction(j)
                            which_candidate(j)=k;
                            max_pred(j) =candidate(k).prediction(j);
                        end
                    end
                end

%                 %PRINT BEST Candidates on screen / along with ground truth                
%                 yellow = uint8([255 255 0]); % [R G B]; class of yellow must match class of I
%                 labels = net.caffe.labels;
%                 RGB    = img;
%                 for k=1:length(which_candidate)
%                     if max_pred(k)>0.8
%                         k_1=which_candidate(k);
%                         bndbox = candidate(k_1).bndbox;
%                         txtposition = [bndbox.xmin,bndbox.ymin];
%                         value    = [labels(k).name ': ' num2str(max_pred(k))];
%                         RGB = insertText(RGB,txtposition,value,'AnchorPoint','LeftTop');
%                         
%                         shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',yellow);
%                         box = [bndbox.xmin,bndbox.ymin,bndbox.xmax-bndbox.xmin,bndbox.ymax-bndbox.ymin];
%                         RGB = step(shapeInserter,RGB,box);
%                     end
%                 end
%                 imshow(RGB);

                for k=1:length(which_candidate) %1-200
                    if max_pred(k)>=net.activation_threshold
                        k_1=which_candidate(k);
                        bndbox = candidate(k_1).bndbox;
                        fprintf(fileID,'%d %d %1.6f %d %d %d %d\n', i, net.label_correspondences(k), candidate(k_1).prediction(k), bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax);
                        APP_LOG('debug','%d %d %1.6f %d %d %d %d', i, net.label_correspondences(k), candidate(k_1).prediction(k), bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax);
                    end
                end
            end
            fclose(fileID);
            if(strcmp(dataset,'validation'))
                [ap , tp_ret, fp_ret, num_pos_ret]=demo_eval_det(net.val_bboxes_dir,[dataset '_results.txt']);
                tp_ret=tp_ret';
                fp_ret=fp_ret';
                num_pos_ret=num_pos_ret';
                for i=1:200
                    net.percentage(i,1).name=net.caffe.labels(find(net.label_correspondences==i)).name;
                    net.percentage(i,1).ap=100*ap(i);
                    net.percentage(i,1).tp=tp_ret(i);
                    net.percentage(i,1).fp=fp_ret(i);
                    net.percentage(i,1).tp_gt=100*tp_ret(i)/num_pos_ret(i);
                end
            end
        end
    end    
end