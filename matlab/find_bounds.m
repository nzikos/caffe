%% DEPRECATED
%% WAS USED TO FIND THE CROPPING BOUNDS ON PROJECTIONS THROUGH EMPTY SPACE DENSITIES
% function [bound_1, bound_2,x1_bound,x2_bound] = find_bounds(x,k1,siz)
%     
%     x_distr_tmp=int16(tabulate(x));
% 
%     x_distr=int32(zeros(siz,2));
%     x_distr(:,1)=1:siz;
%     x_distr(x_distr_tmp(:,1),2)=x_distr_tmp(:,2);
%     
%     x_der=int32(zeros(siz,1));
%     x_2nd_der=int32(zeros(siz,1));
%     for i=2:8
%         x_der(i:end) = x_der(i:end) + x_distr(i:end,2)-x_distr(1:end-i+1,2);
%     end
%     for i=2:8
%         x_2nd_der(i:end) = x_2nd_der(i:end) + x_der(i:end)-x_der(1:end-i+1);
%     end
% 
%     
%     %find first maximum
%     [~,x1_bound]           = max(x_2nd_der(1:k1));   
%     x1_bound=x1_bound-4;
%     if x1_bound<1
%         x1_bound=1;
%     end
%     
%     [~,x2_bound]           = max(x_2nd_der(k1+1:end));
%     x2_bound=x2_bound+k1-1-4;
%             
%     bound_1 =round(x1_bound/2);
%     bound_2 =round(x2_bound + (siz - x2_bound)/2);
% end