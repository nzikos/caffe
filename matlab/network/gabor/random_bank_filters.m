function random_filters = random_bank_filters (x,y,input_num,filters_num)
    random_filters = zeros(x,y,input_num,filters_num); 
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
        diff_g1=fspecial('gaussian', [x y], sigma1);
        diff_g2=fspecial('gaussian', [x y], sigma1);
%       diffGausianFiltersBank(:,:,i) = fspecial('gaussian', [x y], sigma1)-fspecial('gaussian', [x y], sigma2);
        diffGausianFiltersBank(:,:,i) =diff_g1-diff_g2;
    end
    for i=1:input_num
        random_filters(:,:,i,1:gabor_num)=gaborFiltersBank;
        random_filters(:,:,i,(gabor_num+1):end)=diffGausianFiltersBank;
    end
%     %% plot filters 
%     figure('NumberTitle','Off','Name','Random Filters generation');
% for i = 1:4
%     for j = 1:(filters_num/4)        
%         subplot(4,8,(i-1)*8 + j);
%         imshow(real(random_filters(:,:,(i-1)*(filters_num/4) + j)),[]);
%     end
% end
%     %% 
%     fprintf('Gabor are the first = %d filters ',gabor_num );
end