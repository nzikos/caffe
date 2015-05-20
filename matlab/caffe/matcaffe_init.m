function  matcaffe_init(use_gpu, model_def_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(model_def_file)
  % By default use imagenet_deploy
  error('You need a model_def_file (the prototxt)');
end

if caffe('is_initialized') == 1
    caffe('reset');
end
if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('Could not find the prototxt');
end
% load network in train phase
caffe('init', model_def_file,'train')
% set to use GPU or CPU
if use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end
