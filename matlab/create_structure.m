net_struct=NET_STRUCTURE(pwd);

net_struct.set_batch_size('train',128);
net_struct.set_batch_size('validation',128);
net_struct.set_batch_size('test',128);
net_struct.set_input_object_dims(227,227,3);

%LAYER-1
net_struct.add_layer('Convolution' ,{96,11,4,0},{[1,2],{true,[1,2]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('PReLU'       ,{false},{[1 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-2
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{256,5,1,2},{[1,2],{true,[1,2]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('PReLU'       ,{false},{[1 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-3
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{384,3,1,1},{[1,2],{true,[1,2]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('PReLU'       ,{false},{[1 0]},{'constant',0.25});
%LAYER-4
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{384,3,1,1},{[1,2],{true,[1,2]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('PReLU'       ,{true},{[1 0]},{'constant',0.25});
%LAYER-5
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{256,3,1,1},{[1,2],{true,[1,2]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('PReLU'       ,{false},{[1 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-6
net_struct.add_layer('InnerProduct',{4096},{[1,1],{true,[1,1]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('Dropout'     ,{0.5});
net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-7
net_struct.add_layer('InnerProduct',{4096},{[1,1],{true,[1,1]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('Dropout'     ,{0.5});
net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-8
net_struct.add_layer('InnerProduct',{200},{[1,1],{true,[1,1]}},{{'gaussian',[0 0.01]},{'constant',1}});
net_struct.add_layer('Output'      ,{'SoftmaxWithLoss'});

net_struct.validate_structure();
net_struct.create_prototxt('train')
%net_struct.create_prototxt('validation')
net_struct.create_prototxt('test')