
function [laKMM, laMM, BiGraph, A, OBJ, Ah, laKMMh,S,P,W] = KMM_mmconv(X, c, m, k)
% [laKMM, laMM, BiGraph, Anc, ~, ~, ~]= KMM(X', c, m,k) : K-Multiple-Means
% Input:
%       - X: the data matrix of size nFea x nSmp, where each column is a sample
%               point
%       - c: the number of clusters
%       - m: the number of multiple means(MM)
%       - k: the number of neighbor points
% Output:
%       - laKMM: the cluster assignment for each point
%       - laMM: the sub-cluster assignment for each point
%       - BiGraph: the matrix of size nSmp x nMM
%       - A: the multiple means matrix of size nFea x nMM
%       - Ah: the history of multiple means
%       - laKMMh: the history of cluster assignment for each point
% Requre:
%       CSBG.m
% 		meanInd.m
% 		ConstructA_NP.m
% 		EProjSimplex_new.m
% 		svd2uv.m
% 		struG2la.m
%       eig1.m
% Usage:
%       % X: d*n
%       [laKMM, laMM, AnchorGraph, Anchors, ~, ~, ~]= KMM(X', c, m,k) ;
% Reference:
%
%	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
%   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
%   on Knowledge Discovery and Data Mining (KDD â€?19), August 4â€?8, 2019, Anchorage, AK, USA.
%
%   version 1.0 --May./2019 
%
%   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)
if nargin < 4
    if m<6
        k=c-1;
    else
        k=5;
    end      
end
Ah=[];
laKMMh=[];
Iter=15;
OBJ=[]; 
n=size(X{1},2);
viewNum = size(X,2)
method=1; % method for initial seeds, 1:kmeans; 0:random 
opt_conv=1; % option for convergence, 1:sub prototypes; 0:partiton of subclusters

% StartIndZ: before MM update
if method ==0
    StartIndZ=randsrc(n,1,1:m);
else
    for i = 1 : viewNum
        StartIndZ{i}=kmeans(X{i}',m);
    end
end
for i = 1 : viewNum
    BiGraph{i} = ones(n,m);
    A{i} = meanInd(X{i}, StartIndZ{i},m,BiGraph{i});
    [laKMM{i}, laMM{i}, BiGraph{i}, isCov, obj, ~,gamma] = CSBG(X{i}, c, A{i}, k);
end
% [y1,y2, P, Obj1,W] = FSBGC_test_new(X,A,BiGraph,c,m);
% fprintf('time:%d,obj:%d\n',ti,obj)
iter=1;
while(iter<Iter)
    iter = iter +1;
    if isCov
        laKMMh=[laKMMh laKMM];
        Ah=[Ah A];
        OBJ=[OBJ obj];
        if opt_conv==1
            Dis = 0;
            distXt = 0;
            for i = 1 : viewNum
                StartIndZ{i}=laMM{i}; 
                A_old{i} = A{i};
                A{i} = meanInd(X{i}, StartIndZ{i}, m, BiGraph{i});
                Dis = sqdist(A_old{i},A{i}); % O(ndm)
                distXt = Dis + distXt;                
            end
            di = min(distXt, [], 2);
            if norm(di)<1e-4
                fprintf('means converge\n')
                break;
            end
        else            
            if (all(StartIndZ{1}==laMM{1}))
                fprintf('partition converge\n')
                break;
            else
                for i = 1 : viewNum
                    StartIndZ{i}=laMM{i}; 
                    A{i} = meanInd(X{i}, StartIndZ{i},m,BiGraph{i});
                end
            end                   
        end
        for i = 1 : viewNum
            [laKMM{i}, laMM{i}, BiGraph{i}, isCov, obj, ~,gamma] = CSBG(X{i}, c, A{i}, k);   
        end
    else
        if method ==0
            for i = 1 : viewNum
                StartIndZ=randsrc(n,1,1:m);
            end
        else
            for i = 1 : viewNum
                StartIndZ{i}=kmeans(X{i}',m);
            end 
        end
        for i = 1 : viewNum
            BiGraph{i} = ones(n,m);
            A{i} = meanInd(X{i}, StartIndZ{i},m,BiGraph{i});
            [laKMM{i}, laMM{i}, BiGraph{i}, isCov, obj, ~,gamma] = CSBG(X{i}, c, A{i}, k);
        end
    end
fprintf('loop:%d\n',iter);
end

% BiGraphAll = 0;
% for i = 1 : viewNum
%     BiGraph = full(BiGraph);
%     BiGraphAll = BiGraph{i} + BiGraphAll;
% end

% BiGraphAll1{1} = BiGraphAll;
[~,~, P, ~,W,S] = SMKM_P(X,A,BiGraph,c,m);
[laKMM,~] = kmeans(P,c);
% laKMM = y1;

% BiGraphAll = 0;
% for i = 1 : viewNum
%     BiGraph = full(BiGraph);
%     BiGraphAll = BiGraph{i} + BiGraphAll;
% end
% 
% [laKMM,~] = kmeans(BiGraphAll,c);
    
