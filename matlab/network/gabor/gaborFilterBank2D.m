function gaborArray = gaborFilterBank2D(u,v,m,n,gr,freq,gai,psi)

% GABORFILTERBANK generates a custum Gabor filter bank. 
% It creates a u by v array, whose elements are m by n matries; 
% each matrix being a 2-D Gabor filter.
% 
% 
% Inputs:
%       u	:	No. of scales (usually set to 5) 
%       v	:	No. of orientations (usually set to 8)
%       m	:	No. of rows in a 2-D Gabor filter (an odd integer number usually set to 39)
%       n	:	No. of columns in a 2-D Gabor filter (an odd integer number usually set to 39)
% 
% Output:
%       gaborArray: A u by v array, element of which are m by n 
%                   matries; each matrix being a 2-D Gabor filter   
% 
% 
% Sample use:
% 
% gaborArray = gaborFilterBank(5,8,39,39);
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "Identification Using 
%   Encrypted Biometrics," Computer Analysis of Images and Patterns, 
%   Springer Berlin Heidelberg, pp. 440-448, 2013.
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       I WILL APPRECIATE IF YOU CITE OUR PAPER IN YOUR WORK.

% MODIFIED BY: PAPAVASILEIOU LAMPROS - KATSILEROS PETROS
% DATE       : 1/6/2015
% FOR        : vision team - AUTH



%% Create Gabor filters
% Create u*v gabor filters each being an m*n matrix

gaborArray = zeros(m,n,u);
fmax = gai;
gama = sqrt(2);
eta = 2*sqrt(2);

offset = m/4;
 
for i = 1:u
    
    fu = fmax/((sqrt(2))^(i-1));
    alpha = fu/gama;
    beta = fu/eta;
    
    for j = 1:v
%%
    psi= datasample([0;-pi/2],1);
   
    if psi == 0
       freq=1;
    else
       freq=0.5;
    end
    offset_x = -offset + (rand(1) * 2 * offset);
    offset_y = -offset + (rand(1) * 2 * offset);
    tetav= rand(1) * (pi/2);
%%         
          
    gFilter = zeros(m,n);

    for x = 1:m
        for y = 1:n
            xprime = (x-((m+1)/2))*cos(tetav)+(y-((n+1)/2))*sin(tetav)+offset_x;
            yprime = -(x-((m+1)/2))*sin(tetav)+(y-((n+1)/2))*cos(tetav)+offset_y;
            gFilter(x,y) = (fu^2/(pi*gama*eta))*exp(-((alpha^2)*(xprime^2)+(beta^2)*(yprime^2)))*exp(1i*(freq*pi*fu*xprime+psi));
        end
    end
    gaborArray(:,:,j) = real(gFilter);
        
    end
end


%% Show Gabor filters

% Show magnitudes of Gabor filters:
if gr
% figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j);        
%         imshow(abs(gaborArray{i,j}),[]);
%     end
% end

% Show real parts of Gabor filters:
figure('NumberTitle','Off','Name','Real parts of Gabor filters');
for i = 1:4
    for j = 1:(v/4)        
        subplot(4,8,(i-1)*8 + j);
        imshow(real(gaborArray(:,:,(i-1)*(v/4) + j)),[]);
    end
end
end