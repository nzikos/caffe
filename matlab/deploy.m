clear all;

meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');


net = deployed_model(fullfile(cache,'best_model.mat'));
net.caffe.set_use_gpu(1);
net.caffe.set.device(0);
net.caffe.init('test');
net.caffe.reset_object_input('test',2);
net.caffe.set_phase('test');
%net.load_test_imdb(fullfile(imdb,'test'),'.JPEG');
net.load_validation_metadata(fullfile(cache,'meta.mat'));

net.set_output_directory(fullfile(cache,'TEST_RESULTS'));
net.set_overlap(0.1); %values close to 1 lead to heavy overlap
net.set_segments_bounds([2 8]);

x=net.compute('validation');
%load('ILSVRC2014_train_00005977_1.mat');
%object.data =imresize(object.data,[227 227]);
%image(:,:,:,1) = single(object.data)/255;

%x=imread('skilos.JPEG');
%x=imread('mitsos.jpg');
%x=imread('molivia.JPEG');
% x=imread('elena.jpg');
%  y=imresize(x,[227 227]);%, 'bilinear', 'antialiasing', false);
%  image(:,:,:,1) = single(y)/255;
%  image(:,:,:,2) = flip(image(:,:,:,1),2);
%  batch{1}=image;
%  net.caffe.set.input(batch);
%  net.caffe.action.forward();
%  res=net.caffe.get.output();
%  mean_res = (res{1}(1,1,:,1) + res{1}(1,1,:,2)) / 2 ;
% %  %ΤΟΠ-1
% %  [argmax, pos]=max(mean_res);
% %  APP_LOG('info','TOP-1 | Class: %s | %f',net.caffe.labels(pos).name,argmax);
% %ΤΟΠ-κ
% k=5;
% [~,sorted_prediction_ids] = sort(mean_res,3,'descend');
% for i=1:k
%     x=sorted_prediction_ids(i);
%     APP_LOG('info','TOP-%d | Class: %s | %1.2f%%',i,net.caffe.labels(x).name,mean_res(x)*100);
% end
