function [ L2,A2,B2] = lvl_arrays(level,L,A,B)
dim=2^level; %dim*dim=number of submatrices
i=1:dim;
dim1(i)=length(L)/dim;
L2=mat2cell(L,dim1,dim1);
A2=mat2cell(A,dim1,dim1);
B2=mat2cell(B,dim1,dim1);
end

