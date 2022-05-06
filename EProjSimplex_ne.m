function [X] = EProjSimplex_new(V)

%
%% Problem
%
%  min  1/2 || X - V||_F^2
%  s.t. x>=0, 1'x=1
%

[n,d]=size(V);

if issparse(V)
    V1=full(V);
    V1(V1==0)=-1;
    X=sort(V1,2,'descend');
    Xtmp=(cumsum(X,2)-1)*diag(sparse(1./(1:d)));
    X = max(bsxfun(@minus,V1,Xtmp(sub2ind([n,d],(1:n)',sum(X>Xtmp,2)))),0);
    X=sparse(X);
else
    X=sort(V,2,'descend');
    Xtmp=(cumsum(X,2)-1)*diag(sparse(1./(1:d)));
    X = max(bsxfun(@minus,V,Xtmp(sub2ind([n,d],(1:n)',sum(X>Xtmp,2)))),0);
end
