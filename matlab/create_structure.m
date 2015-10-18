clear all;

net.caffe.structure=NET_STRUCTURE(pwd);

net.caffe.structure.set_batch_size('train',64);
net.caffe.structure.set_batch_size('validation',128);
net.caffe.structure.set_batch_size('test',10);
net.caffe.structure.set_input_object_dims(3,227,227);
net.caffe.structure.set_labels_length(1);
net.caffe.structure.add_CONV_layer   ('conv1',96,11,0,4,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu1','relu');
%net.caffe.structure.add_LRN_layer    ('lrn1' ,3,0.0000005,0.75,'within_channel');
net.caffe.structure.add_POOL_layer   ('pool1','MAX',3,0,2);

net.caffe.structure.add_CONV_layer   ('conv2',256,5,2,1,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu2','relu');
%net.caffe.structure.add_LRN_layer    ('lrn2' ,3,0.0000005,0.75,'within_channel');
net.caffe.structure.add_POOL_layer   ('pool2','MAX',3,0,2);

net.caffe.structure.add_CONV_layer   ('conv3',384,3,1,1,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu3','relu');

net.caffe.structure.add_CONV_layer   ('conv4',384,3,1,1,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu4','relu');

net.caffe.structure.add_CONV_layer   ('conv5',256,3,1,1,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu5','relu');
net.caffe.structure.add_POOL_layer   ('pool5','MAX',3,0,2);

net.caffe.structure.add_IP_layer     ('fc6'  ,4096,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu6','relu');
net.caffe.structure.add_DROPOUT_layer('drop6',0.5);

net.caffe.structure.add_IP_layer     ('fc7'  ,4096,true,'gaussian',0.01,1,'constant',0,2);
net.caffe.structure.add_ACTIV_layer  ('relu7','relu');
net.caffe.structure.add_DROPOUT_layer('drop7',0.5);

net.caffe.structure.add_IP_layer     ('fc8'  ,200,true,'gaussian',0.01,1,'constant',0,2); 

net.caffe.structure.add_OUTPUT_ERROR_layer('SOFTMAX_LOSS');

net.caffe.structure.validate_structure();
net.caffe.structure.create_prototxt('train')
net.caffe.structure.create_prototxt('validation')
net.caffe.structure.create_prototxt('test')