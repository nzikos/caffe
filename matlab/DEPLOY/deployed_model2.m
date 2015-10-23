classdef deployed_model2 < handle
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
            
            APP_LOG('debug','Setting model weights and bias');
            net.caffe.params   = model.params;
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
                for j=segments_bounds(1):segments_bounds(2)                
                    subregions = subregions + i*j;
                end
            end
            net.total_subregions = subregions;
            APP_LOG('info','Total subregions per image: %d',subregions);
            net.total_objects    = subregions + subregions; %AND THE FLIPPED ones
            APP_LOG('info','Total objects fed into network per image: %d',net.total_objects);
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
            figure;
            
            for i=1:length(img_list)
                APP_LOG('info','Image %d/%d',i,length(img_list));
                img = imread(img_list{i});
                if size(img,3)==1 %B&W IMAGES
                    img = repmat(img,1,1,3);
                end
                imshow(img);
                hold on;
                
                original_size = net.validation_meta(i).size;
                img = im2single(imresize(img,original_size));
                
                input_size = [net.caffe.structure.object_height,net.caffe.structure.object_width];
                
                cnd_id =1;
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
                                image{1}(:,:,:,1) = imresize(tmp,input_size,'bilinear', 'antialiasing', false);
                                image{1}(:,:,:,2) = flip(image{1}(:,:,:,1),2);
                                net.caffe.set.input(image);
                                net.caffe.action.forward();
                                ret=net.caffe.get.output();
                                prediction = squeeze(mean(ret{1},4));
                                
                                candidate(cnd_id).bndbox = bndbox;
                                candidate(cnd_id).prediction = prediction;
                                cnd_id=cnd_id+1;
                                rectangle('Position',[bndbox.xmin,bndbox.ymin,bndbox.xmax-bndbox.xmin,bndbox.ymax-bndbox.ymin],'EdgeColor',[segments_x/net.segments_bounds(2) segments_y/net.segments_bounds(2) 0],'LineWidth',1);
                                drawnow limitrate;
                            end
                        end
                    end
                end
                hold off;
                close;
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
                %PRINT BEST Candidates on screen / along with ground truth
                yellow = uint8([255 255 0]); % [R G B]; class of yellow must match class of I
                labels = net.caffe.labels;
                RGB    = img;
               for k=1:length(which_candidate)
                    if max_pred(k)>0.9
                        k_1=which_candidate(k);
                        bndbox = candidate(k_1).bndbox;
                        txtposition = [bndbox.xmin,bndbox.ymin];
                        value    = [labels(k).name ': ' num2str(max_pred(k))];
                        RGB = insertText(RGB,txtposition,value,'AnchorPoint','LeftTop');
                        
                        shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',yellow);
                        box = [bndbox.xmin,bndbox.ymin,bndbox.xmax-bndbox.xmin,bndbox.ymax-bndbox.ymin];
                        RGB = step(shapeInserter,RGB,box);
                    end
               end
%                 [~,max_of_max]=max(max_pred);
%                 k_1 = which_candidate(max_of_max);
%                 bndbox = candidate(k_1).bndbox;
%                 txtposition = [bndbox.xmin,bndbox.ymin];
%                 value    = [labels(max_of_max).name ': ' num2str(max_pred(max_of_max))];
%                 RGB = insertText(RGB,txtposition,value,'AnchorPoint','LeftTop');
%                 
%                 shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',yellow);
%                 box = [bndbox.xmin,bndbox.ymin,bndbox.xmax-bndbox.xmin,bndbox.ymax-bndbox.ymin];
%                 RGB = step(shapeInserter,RGB,box);
                figure;
                imshow(RGB);
                
                [~,filename,~] = fileparts(img_list{i});
                meta.size = net.validation_meta(i).size;
                meta.objs = net.validation_meta(i).objs;
                APP_LOG('info','saved: %s',fullfile(net.output_directory,[filename '.mat']));
                save(fullfile(net.output_directory,[filename '.mat']),'meta','candidate');
            end
        end
    end    
end