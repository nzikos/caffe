% This function produces filters_num filters some of them Gabor and some
% Dog (Difference of Gaussians)

% AUTHOR: PAPAVASILEIOU LAMPROS
% AUTHOR: KATSILEROS PETROS
% DATE:   1/6/2015
% FOR:    vision team - AUTH


function filters_struct = random_bank_filters (x,y,filters_num,channels)
    random_filters = zeros(x,y,filters_num);
    
    gabor_num=randi(filters_num);
    diffg_num=filters_num-gabor_num;
    
    %% sigma range
    sigmalow=0;
    sigmahigh=x/3;

    gaborFiltersBank = zeros(x,y,gabor_num);
    diffGausianFiltersBank = zeros(x,y,diffg_num);

    
    gaborFiltersBank(:,:,:) = gaborFilterBank2D(1,gabor_num,x,y,0,0,0.8,0);
    
    for i=1:diffg_num
        sigma1=rand(1) * (sigmahigh - (sigmalow)) + (sigmalow);
        sigma2=rand(1) * (sigmahigh - (sigmalow)) + (sigmalow);
        diffGausianFiltersBank(:,:,i) = fspecial('gaussian', [x y], sigma1)-fspecial('gaussian', [x y], sigma2);
    end
    
    random_filters(:,:,1:gabor_num)=gaborFiltersBank;
    random_filters(:,:,(gabor_num+1):end)=diffGausianFiltersBank;
    
    %% plot filters 
%     figure('NumberTitle','Off','Name','Random Filters generation');
% for i = 1:4
%     for j = 1:(filters_num/4)        
%         subplot(4,8,(i-1)*8 + j);
%         imshow(real(random_filters(:,:,(i-1)*(filters_num/4) + j)),[]);
%     end
% end
    %% 
%     fprintf('Gabor are the first = %d filters \n',gabor_num );
filters_struct = zeros(size(random_filters,1),size(random_filters,2),channels,filters_num);
for i=1:filters_num
    for j=1:channels
%         filters_struct{i,1} = zeros(size(random_filters,1),size(random_filters,2),channels);
        filters_struct(:,:,j,i) = random_filters(:,:,i);
    end
end
end