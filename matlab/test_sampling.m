clear all;

K=1000;
N=512;

v=zeros(1,K);

maximum_concurrent_same_samples = 10;
p = zeros(1,maximum_concurrent_same_samples);
p_2 = zeros(1,maximum_concurrent_same_samples);

num_samples=10000;
m_fig = animatedline('color',[110 181 254]/255);
for l=1:maximum_concurrent_same_samples
    for i=1:num_samples
        w_uniform=ones(1,K)/K;
%        w_uniform(2)=0.5;
%        w_uniform=w_uniform/sum(w_uniform);
%         for n=1:N
%             v(n)=randsample(K,1,true,w_uniform);
%             w_uniform(v(n))=w_uniform(v(n))*w_uniform(v(n));
%         end
%       v=randsample(K,N,true,w_uniform);
        v=randsampleWR(K,N,w_uniform);
        v_unique=unique(v);
        %   for a single class
        %    for j=length(v_unique):-1:1
        %        how_many_samples(j) = sum(v==v_unique(j));
        %    end
        %     a=find(v_unique==1);
        %     if ~isempty(a)
        %         if how_many_samples(a)>=2
        %             p_2 = p_2 +1;
        %         end
        %     end
        %   for all the classes
        for j=length(v_unique):-1:1
            how_many_samples(j) = sum(v==v_unique(j));            
        end
        a=find(v_unique==2);
        if ~isempty(a)
            p_2(l)=p_2(l)+how_many_samples(a);
        end
        for j=length(v_unique):-1:1
            how_many_samples = sum(v==v_unique(j));
            if how_many_samples>=l
                p(l)=p(l)+1;
                break;
            end
        end
    end
    p(l)=(p(l)/num_samples)*100;
    p_2(l)=(p_2(l)/(num_samples*N))*100
    addpoints(m_fig,l,p(l));
    drawnow
end
%p_2 = (p_2 / num_samples)*100