caffe('reset');
caffe('set_mode_gpu');
caffe('init',fullfile(pwd,'caffe','prototxt','tst_dummy_quick.prototxt'),'train');

x=caffe('get_weights');
for i=1:8
    for j=1:2
        x(i).weights{j}  = x(i).weights{j}+1;
    end
end
caffe('set_weights',x);

y=caffe('get_weights');
for i=1:8
    for j=1:2
        y(i).weights{j}  = y(i).weights{j}+1;
    end
end
caffe('set_weights',y);