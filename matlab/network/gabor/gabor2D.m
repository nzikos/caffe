% kernel are the kernel's dimensions of the filter (15x15)
% example: gabor2D(15) -> 32 filters of size 15x15 for 3 color channels (rgb)

function [gaborFiltersBank] = gabor2D(kernel)
    gaborFiltersBank = zeros(kernel,kernel,3,32);
    % Make gabor filters bank
    for i=1:3
        gaborFiltersBank(:,:,i,1:8) = gaborFilterBank2D(1,8,kernel,kernel,0,0.5,0.6,-(pi/2));
        gaborFiltersBank(:,:,i,9:16) = gaborFilterBank2D(1,8,kernel,kernel,0,1,0.6,0);
        gaborFiltersBank(:,:,i,17:24) = gaborFilterBank2D(1,8,kernel,kernel,0,1,0.5,0);
        gaborFiltersBank(:,:,i,25:32) = gaborFilterBank2D(1,8,kernel,kernel,0,1,0.3,0);
    end
end