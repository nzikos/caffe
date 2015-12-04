model          = 'best_model_uni_with_replacement.mat';
model_filepath = fullfile(pwd,model);

if ~exist(model_filepath,'file')
    APP_LOG('info','Downloading model');
    websave('temp','https://www.dropbox.com/s/pn9wghlg1gwmi75/best_model_uni_with_replacement.mat?dl=1',weboptions('Timeout',Inf));
    movefile('temp',model);
end

net = deployed_model3(model_filepath,-1);

img = imread('printer.jpg');
%img = imread('laptop.jpg');
%img  = permute(img, [2 1 3]);
imshow(img)
[class,conf]=net.predict(img);

for i=1:5
    APP_LOG('info','%s %f%%',class{i},conf(i)*100);
end