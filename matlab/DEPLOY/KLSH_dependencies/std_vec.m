function [ vec] = std_vec(L,A,B,Lw,Aw,Bw)
L=cellfun(@std2,L);
A=cellfun(@std2,A);
B=cellfun(@std2,B);
vec=[L(:)*Lw;A(:)*Aw;B(:)*Bw];
end

