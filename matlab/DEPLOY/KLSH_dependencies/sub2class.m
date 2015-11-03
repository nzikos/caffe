function [ classHist ] = sub2class( scores ,connections,weights)
%creates the classHist(200 classes) using CategoryHist (569 subcategories);
classHist=zeros(200,1);

for i=1:200
    %counter=0;
    for j=1:569
        if (i==connections(j).classIndex)
           %counter=counter+1; 
           classHist(i)= classHist(i)+scores(j);
        end
    end
    classHist(i)=classHist(i)/weights(i);
end

%normalize
classHist=classHist./sum(classHist);