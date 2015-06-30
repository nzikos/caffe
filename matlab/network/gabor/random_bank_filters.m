% This function produces filters_num filters some of them Gabor and some
% Dog (Difference of Gaussians)

% AUTHOR: PAPAVASILEIOU LAMPROS
% AUTHOR: KATSILEROS PETROS
% DATE:   1/6/2015
% FOR:    vision team - AUTH

% MODIFIED TO SUPPORT CAFFE STRUCTURE as [kernel,kernel,filters,outputs]
% Filters are re-shuffled per input with randperm
% Provos Alexis

function random_filters = random_bank_filters (x,y,filters_num,layer_outputs)
    random_filters = zeros(x,y,filters_num,layer_outputs); 
    for i=1:layer_outputs
        tmp_filters    = zeros(x,y,filters_num);
        gabor_num=randi(filters_num);
        diffg_num=filters_num-gabor_num;
    
        %% sigma range
        sigma = x/3;

        gaborFiltersBank = zeros(x,y,gabor_num);
        diffGausianFiltersBank = zeros(x,y,diffg_num);
    
       gaborFiltersBank(:,:,:) = gaborFilterBank2D(1,gabor_num,x,y,0,0,0.8,0);
    
        for j=1:diffg_num
            sigma1=rand(1) * sigma;
            sigma2=rand(1) * sigma;
            diff_g1=fspecial('gaussian', [x y], sigma1);
            diff_g2=fspecial('gaussian', [x y], sigma2);
            diffGausianFiltersBank(:,:,j) =diff_g1-diff_g2;
        end
    
        tmp_filters(:,:,1:gabor_num)=gaborFiltersBank;
        tmp_filters(:,:,(gabor_num+1):end)=diffGausianFiltersBank;

        % Random permutation of Gabor and DoG filters
        tmp_filters = tmp_filters(:,:,randperm(size(tmp_filters,3)));
        
        % prepare filters for caffe
        random_filters(:,:,:,i) = tmp_filters;
        fprintf('Gabor are the first = %d filters.\n',gabor_num );        
    end
    %% plot filters - may be bugged (remove if not)
%     figure('NumberTitle','Off','Name','Random Filters generation');
%     for i = 1:4
%         for j = 1:(filters_num/4)        
%             subplot(4,8,(i-1)*8 + j);
%             imshow(real(tmp_filters(:,:,(i-1)*(filters_num/4) + j)),[]);
%         end
%     end
    %% 
end