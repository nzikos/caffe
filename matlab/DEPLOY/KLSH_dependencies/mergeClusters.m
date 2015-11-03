function clusters = mergeClusters(clusters, clusters2)

for i=1:length(clusters.exp)
   clusters.exp(i).id = [clusters.exp(i).id ; clusters2.exp(i).id];
   clusters.exp(i).population = [clusters.exp(i).population ; clusters2.exp(i).population];
end