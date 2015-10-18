for i=109:109
out.res(i).probs = double(net.validation_output(i).prob_area);
out.res(i).img = imread(net.validation_meta(i).imdb_cor);
end
out.labels = net.caffe.labels;
%save('out.mat','out','-v7.3');
%clear all;
%if ~exist('out','var')
%    load out;
%end
imshow(out.res(109).img);
figure;
colormap summer;
surfc(out.res(109).probs(:,:,27),'LineWidth',0.5)

img=out.res(109).img;
image = zeros(size(img,1),size(img,2),1);   %# The z data for the image corners
hold on;
%figure;
surf(image,'CData',img,'FaceColor','texturemap','LineStyle','none');
set(gca,'Xdir','reverse','Ydir','reverse')
hold off;
