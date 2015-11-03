function [ vec] = mean_vec(L,A,B,Lw,Aw,Bw)
L=cellfun(@mean2,L);
A=cellfun(@mean2,A);
B=cellfun(@mean2,B);
vec=[L(:)*Lw;A(:)*Aw;B(:)*Bw];
end