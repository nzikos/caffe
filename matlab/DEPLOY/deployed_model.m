classdef deployed_model < handle
    properties 
        caffe;
        validation_meta;
        test_img_list;
                
        stride              = [];
        segments_bounds     = [];
        total_subregions    = [];
        total_objects       = [];
        
        validation_output;
        testing_output;
        
        output_directory;
    end
    methods
        function net = deployed_model(model_path)
            APP_LOG('info','Loading model');
            load(model_path);
            [model_dir,~,~]     = fileparts(model_path);
            
            APP_LOG('debug','Initializing caffe wrapper');
            net.caffe           = CAFFE(model_dir);
            
            APP_LOG('debug','Loading network structure');
            net.caffe.set_structure(model.structure);
            
            APP_LOG('debug','Loading labels');
            net.caffe.labels    = model.labels;
            
            APP_LOG('debug','Setting model weights');
            net.caffe.weights   = model.weights;
        end
        function net = load_test_imdb(net,directory,extension)
            net.test_img_list = super_get_file_list(directory,extension);
        end
        function net = load_validation_metadata(net,val_meta)
            tmp=load(val_meta);
            net.validation_meta = tmp.data.validation;
        end
        
        function net = set_output_directory(net,dir)
            handle_dir(dir,'create');
            net.output_directory = dir;
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
                subregions = subregions + i^2;
            end
            net.total_subregions = subregions;
            APP_LOG('info','Total subregions per image: %d',subregions);
            net.total_objects    = subregions + subregions; %AND THE FLIPPED ones
            APP_LOG('info','Total objects fed into network per image: %d',net.total_objects);
        end

        function segments = compute(net,dataset)
            switch(dataset)
                case 'validation'
                    img_list = {net.validation_meta.imdb_cor};
                case 'test'
                    img_list = net.test_img_list;
                otherwise
                    APP_LOG('last_error','Unknown dataset. Use validation/test');
            end
            %figure;
            
%            for i=1:length(img_list)
                for i=109:109
%                 try
                    APP_LOG('info','Image %d/%d',i,length(img_list));
                    img = imread(img_list{i});
                    if size(img,3)==1 %B&W IMAGES
                        img = repmat(img,1,1,3);
                    end
                    original_size = net.validation_meta(i).size;
                    img = imresize(img,original_size);

                    input_size = [net.caffe.structure.object_height,net.caffe.structure.object_width];
                    segments = uint8(zeros(input_size(1),input_size(2),3,net.total_objects));
                    
                    prob_layer=single(zeros(original_size(1),original_size(2),length(net.caffe.labels)));                    
                    prob_layer_mean_counter = single(zeros(original_size(1),original_size(2)));                    
                    for j=net.segments_bounds(1):net.segments_bounds(2)
                        scaled_img      = imresize(img,[(j-1)*net.stride(1)+input_size(1),(j-1)*net.stride(2)+input_size(2)]);
                        scaled_size     = size(scaled_img);
                        inner_counter   = j^2;
                        for segment_x=j-1:-1:0
                            for segment_y=j-1:-1:0
                                xmin = segment_x*net.stride(1) + 1;
                                xmax = xmin + input_size(1)  - 1;
                                ymin = segment_y*net.stride(2) + 1;
                                ymax = ymin + input_size(2)  - 1;                              
                                %find original dims representation for each segment
                                this_dims(inner_counter).xmin = 1+ floor((xmin-1)*(original_size(1)/scaled_size(1)));
                                this_dims(inner_counter).xmax = floor((xmax)*(original_size(1)/scaled_size(1)));
                                this_dims(inner_counter).ymin = 1+floor((ymin-1)*(original_size(2)/scaled_size(2)));
                                this_dims(inner_counter).ymax = floor((ymax)*(original_size(2)/scaled_size(2)));

                                image{1}(:,:,:,1) = im2single(scaled_img(xmin:xmax,ymin:ymax,:));
                                image{1}(:,:,:,2) = flip(image{1}(:,:,:,1),2);
                                net.caffe.set.input(image);
                                net.caffe.action.forward();
                                ret=net.caffe.get.output();
                                prediction = mean(ret{1},4);
                                this_preds{inner_counter,1} = prediction;
% [~,sorted_prediction_ids] = sort(prediction,3,'descend');
% imshow(scaled_img(xmin:xmax,ymin:ymax,:));hold on;
% for ll=1:3
%      x=sorted_prediction_ids(ll);
%      APP_LOG('info','TOP-%d | Class: %s | %1.2f%%',ll,net.caffe.labels(x).name,prediction(x)*100);
% end                                

                                 xmin = this_dims(inner_counter).xmin;
                                 xmax = this_dims(inner_counter).xmax;
                                 ymin = this_dims(inner_counter).ymin;
                                 ymax = this_dims(inner_counter).ymax;
                                 prob_layer(xmin:xmax,ymin:ymax,:)            = prob_layer(xmin:xmax,ymin:ymax,:) + repmat(prediction,xmax-xmin+1,ymax-ymin+1,1);
                                 prob_layer_mean_counter(xmin:xmax,ymin:ymax) = prob_layer_mean_counter(xmin:xmax,ymin:ymax) +1;
                                inner_counter=inner_counter-1;
                            end
                        end
%                        net.validation_output(i).seg_depth(j).dims = this_dims;
%                        net.validation_output(i).seg_depth(j).preds = this_preds;
%                        seg_depth(j).dims = this_dims;
%                        seg_depth(j).preds = this_preds;
                    end
                    
                     for class=1:length(net.caffe.labels)
                         prob_layer(:,:,class) = prob_layer(:,:,class) ./ prob_layer_mean_counter;
                     end
                    
                    net.validation_output(i).prob_area = prob_layer;
%                     [~,filename,ext] = fileparts(img_list{i});
%                     out.name = strcat(filename,ext);
%                     out.metadata = net.validation_meta(i);
% %                    out.seg_depth= seg_depth;
% %                    out.prob_layer = prob_layer;
%                     out.predictions = squeeze(prediction);
%                     APP_LOG('info','saved: %s',fullfile(net.output_directory,[filename '.mat']));                    
%                     save(fullfile(net.output_directory,[filename '.mat']),'out','-v7.3');

%                     clear out;
%                 catch err
%                     APP_LOG('warning','%s',err.message);
%                 end
            end
        end
    end    
end


%                imshow(img);
%                hold on;
%                for j=1:length(net.validation_meta(i).objs)
%                    xmin=net.validation_meta(i).objs(j).bndbox.xmin;
%                    xmax=net.validation_meta(i).objs(j).bndbox.xmax;
%                    ymin=net.validation_meta(i).objs(j).bndbox.ymin;
%                    ymax=net.validation_meta(i).objs(j).bndbox.ymax;
%                    rectangle('Position',[xmin, ymin, xmax-xmin, ymax-ymin],'EdgeColor','r','LineWidth',2);
%                end
%                hold off;