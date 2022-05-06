function [ nc_labels ] = n2nc( labels ,numLabels)
%N2NC rows x 1 to rows x unique(labels) , for each label in labels, nc_labels(the row of label, label) = 1
%   labels = [1;2] => new_labels = [[1,0],[0,1]]

n = length(labels);
ylist = unique(labels);
if nargin<2
    numLabels= length(ylist);
end
nc_labels = zeros(n, numLabels);

len=numLabels;
if len>length(ylist)
    len=length(ylist);
end

for i = 1:len
    c = ylist(i);
    nc_labels(labels == c, i) = 1;
end

