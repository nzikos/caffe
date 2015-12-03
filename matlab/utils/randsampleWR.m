%randsampleWR Same as randsample with weights but Without Replacement

%% Originally written by ROY
% http://www.mathworks.com/matlabcentral/newsreader/view_thread/141124

%% AUTHOR: PROVOS ALEXIS
%  DATE:   20/11/2015
%  FOR:    VISION TEAM - AUTH

function v=randsampleWR(n,k,w)
% If asked samples (k) are more than number of potential samples (n)
if k>n
    APP_LOG('last_error','Can not perform random sampling without replacement with batch size bigger than number of categories');
end
% if asked samples (k) are less than number of potential samples (n)
% perform random sampling without replacement
v=zeros(1,k);    
if k<n
    if all(w==w(1))
        v(1:k)=randperm(n,k);
    else
        for i=1:k
            v(i)=randsample(n,1,true,w);
            w(v(i))=0;
            w=w./sum(w); %isnt it performed internally?
        end
    end
end
if k==n
    v=1:n;
end
end