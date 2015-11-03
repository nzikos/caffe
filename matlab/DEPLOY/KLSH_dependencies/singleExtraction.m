function [ testFeatures ] = singleExtraction( query_img,w )
%features_param={{'hist0_20',w(1)/2048},{'mean_1',w(2)/255},{'std_1',w(3)/64},{'mean_2',w(4)/255},{'std_2',w(5)/64}};

%query= query path
Lw=0.5;Aw=0.25;Bw=0.25;

% temp=imfinfo(query);
%  if (strcmp(temp.ColorType,'truecolor')||strcmp(temp.ColorType,'grayscale'))
      im=imresize(query_img,[64 64]);
%      if size(im,3)==1
%      im=repmat(im,[1 1 3]);
%      end
% end
[ L,A,B ] = rgb2lab( im );  
F=[imhist(L,20)*Lw;imhist(A,20)*Aw;imhist(B,20)*Bw]*w(1)/2048; %hist0_20
[ L2,A2,B2] = lvl_arrays(1,L,A,B);
F=[F; mean_vec(L2,A2,B2,Lw,Aw,Bw)*w(2)/255];%mean_1
F=[F;std_vec(L2,A2,B2,Lw,Aw,Bw)*w(3)/64];%std_1
[ L2,A2,B2] = lvl_arrays(2,L,A,B);
F=[F;mean_vec(L2,A2,B2,Lw,Aw,Bw)*w(4)/255];%mean_2
F=[F;std_vec(L2,A2,B2,Lw,Aw,Bw)*w(5)/64];%std_2
testFeatures=struct('F',F);
end

