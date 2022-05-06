% function [ y1, y2, alphas, P, Obj1, Obj2, mu, rho, B0, B ] = FSBGC_test( clusters, Params ,scale,projection_dim)
function [ y1, y2,  P, Obj1,W] = FSBGC_test(BiGraph,clusterNum, m, clusters, Params ,scale,projection_dim)


if nargin < 3
    scale=0.7;
end

if nargin < 4
    %projection_dim=2;
    projection_dim=4;
end

%clusterNum=Params.K;

n = size(BiGraph{1},1);
m = size(BiGraph{1},2);
viewNum = size(BiGraph,2)
% [n, m]=size(clusters);

if projection_dim < clusterNum
    projection_dim = clusterNum+10;
end

projection_dim = m;
%% 

for i = 1:viewNum
    
    B{i} = BiGraph{i};
    
end



P_ITERS=100;
Obj1=zeros(1,P_ITERS);


[n,~]=size(B{1});

%��ʼ��P_next����
P_next = rand(n,projection_dim);

P_next_i = zeros(n,projection_dim);
temp = zeros(n,projection_dim);

for i =1:viewNum
   BB_T{i} =  (B{i}'*B{i})^-1*B{i}';
end
 



%�ȸ�����w %�ٸ�����p_next ��ʼ��p_next����W����Ϣ ����Pʱ�Ȳ�Ҫ��R ������ȫ1���� 
    for i = 1:viewNum
        
        W{i} = BB_T{i}*P_next;  

        temp = B{i}*W{i};
%         temp(temp<0) = 0;
%         P_next_i = P_next_i + temp; 
        E = EProjSimplex_ne(temp);
        P_next_i = P_next_i + E;               
    end     
    
P_next = P_next_i;
P_next =P_next/m;   
%��ʼ��P_next�������


lambda = 100;

W = {};
    
DPT = diag(sum(P_next));
   
    try
        [U,V,~,ev] = svdfast(P_next*DPT^-0.5, clusterNum);

    catch err
       disp(err); 
    end
    
R = U*V'*DPT^-0.5;
temp_R = lambda/m*R;

%��ʼ����������ز���
for iter1=1:P_ITERS
    
    % ԭ����P��ϡ�裬W�ĳ�ʼ��Ϊ�����ˣ�Ȼ��RҲ�����ģ�
    % ���һ����Ҳ�����ģ�max֮��Ͳ�ϡ���ˣ��������Ƿ����ָ�ֵ��R����Pϡ�費��֪��
    % �����Ҷ�P���˴��ڵ���1��Լ�������ܲ���ϡ�����
    % ������У����W�����һЩ���ѡ��һЩ������ֵ������
    % ������R��ʼ����һЩ��ֵ���ٳ�ʼ��P�����߶Գ�ʼ��P��k����Լ��������ֵȡ������
    % �ֻ��߶�P��������Լ����

    % �����Ƕ�P�Ӵ��ڵ���1��Լ�������ܲ���ϡ�����
    
%     P_next_i = zeros(n,projection_dim);
%     temp = zeros(n,projection_dim);    
%     for i = 1:m
%         
%         W{i} = (B{i}'*B{i})^-1*B{i}'*P_next;  
%         temp = B{i}*W{i} + lambda/m*R;
%         temp(temp<0) = 0;              
%         P_next_i = P_next_i + temp; 
%     end        
%         P_next = P_next_i;
%         P_next =P_next/m;
    P_next_i = zeros(n,projection_dim);
    temp = zeros(n,projection_dim);    
    for i = 1:viewNum
        
        W{i} = BB_T{i}*P_next;  
        temp = B{i}*W{i} + temp_R;
        E = EProjSimplex_ne(temp);
        P_next_i = P_next_i + E; 
    end
    P_next = P_next_i;
    P_next =P_next/viewNum;    
    
%     DPT = diag(sum(P_next));    

    DPT = diag(sum(P_next) + eps); 
    
    try
        [U,V,~,ev] = svdfast(P_next*DPT^-0.5, clusterNum);
        U = sqrt(2)/2*U; 
        V = sqrt(2)/2*V;
    catch err
       disp(err); 
    end    
    
    R = U*V'*DPT^-0.5;
    temp_R = lambda/m*R;

    F=[U; V];
%     S=sparse(n+projection_dim,n+projection_dim);
%     %����bipartite Graph����
%     S(1:n,n+1:end)=P_next; 
%     S(n+1:end,1:n)=P_next';
%     %����ǿ��ͨ���� ��ǿ��ͨ��ͼ ÿ��ǿ��ͨ��ͼ���Ǿ۵�һ����
%     [comp_number, y]=graphconncomp(S);
%     y1=y(1:n)';
%     y2=y(n+1:end)';
%     
%     if comp_number ~= clusterNum
%         sprintf('Can not find the correct cluster number: %d', clusterNum)
%     else
%         sprintf('Have fund the correct cluster number: %d', clusterNum)
%     end
    
    fn1 = sum(ev(1:clusterNum));
    %fn2 = sum(ev(1:clusterNum+1));
 
    %jinjigenggai1 214 
    fn2 = sum(ev(1:clusterNum));
    
 %���õ���fn2������ʵ�������������lambda�ļ�Сfn2˥�����������������Ŀ�꺯��������������Բ����� e-ʮ��   
    
    if fn1 < clusterNum-0.0000001
        lambda = 2*lambda;
    else
        if fn2 > clusterNum+1-0.0000001
            lambda = lambda/2;
        end
    end    

%Ŀ�꺯�� temp - temp2
    temp = 0; 
    temp2 = 0;
    
    for i = 1:viewNum
        temp = temp + F22norm(B{i}*W{i}-P_next);
    end
    

    temp2 = 2*lambda*trace(U'*P_next*DPT^-0.5*V);
    
    Obj1(iter1)=temp - temp2;

%     Obj1(iter1)=F22norm(P_next-sum_WB);
    
    if iter1 >1
        change=abs((Obj1(iter1)-Obj1(iter1-1))/Obj1(iter1));
        if change<1e-8
                S=sparse(n+projection_dim,n+projection_dim);
                %����bipartite Graph����
                S(1:n,n+1:end)=P_next; 
                S(n+1:end,1:n)=P_next';
                %����ǿ��ͨ���� ��ǿ��ͨ��ͼ ÿ��ǿ��ͨ��ͼ���Ǿ۵�һ����
                [comp_number, y]=graphconncomp(S);
                y1=y(1:n)';
                y2=y(n+1:end)';
    
                if comp_number ~= clusterNum
                    sprintf('Can not find the correct cluster number: %d', clusterNum)
                else
                    sprintf('Have fund the correct cluster number: %d', clusterNum)
                end
                break;
        end
    end
    
    if iter1 == P_ITERS
             S=sparse(n+projection_dim,n+projection_dim);
             %����bipartite Graph����
             S(1:n,n+1:end)=P_next; 
             S(n+1:end,1:n)=P_next';
             %����ǿ��ͨ���� ��ǿ��ͨ��ͼ ÿ��ǿ��ͨ��ͼ���Ǿ۵�һ����
             [comp_number, y]=graphconncomp(S);
             y1=y(1:n)';
             y2=y(n+1:end)';
    
             if comp_number ~= clusterNum
                 sprintf('Can not find the correct cluster number: %d', clusterNum)
             else
                 sprintf('Have fund the correct cluster number: %d', clusterNum)
             end
             break; 
    end
    

    


end
%     folder_now = pwd;
%     path1 = [folder_now '\Analog'];
%     
%     touying = {};

    P=P_next;

%     for i = 1:m
%         
%         iter = i;
%         touying{i} = B{i}*W{i};
%         temp = touying{i};
%         %dlmwrite(['\_time1111.csv'], Result_Time, ',', 0, 0);
%         csvwrite([path1 '\BW_' num2str(i) 'juzhen.csv'],touying{i});
%         csvwrite([path1 '\B_' num2str(i) 'juzhen.csv'],B{i});
%         csvwrite([path1 '\W_' num2str(i) 'juzhen.csv'],W{i});
%         %dlmwrite([path1 '\' iter '_touying.csv'], temp, ',', 0, 0);
%     end

end


% %%
% function [newL2, c] = bestMap(L1,L2)
% %bestmap: permute labels of L2 match L1 as good as possible
% %   [newL2] = bestMap(L1,L2);
% 
% %===========
% L1 = L1(:);
% L2 = L2(:);
% if size(L1) ~= size(L2)
%     error('size(L1) must == size(L2)');
% end
% L1 = L1 - min(L1) + 1;      %   min (L1) <- 1;
% L2 = L2 - min(L2) + 1;      %   min (L2) <- 1;
% %===========    make bipartition graph  ============
% nClass = max(max(L1), max(L2));
% G = zeros(nClass);
% for i=1:nClass
%     for j=1:nClass
%         %%ͬһ��������L1��Ԥ��Ϊi�ಢ����L2��Ԥ��Ϊj��ĸ���
%         G(i,j) = length(find(L1 == i & L2 == j));
%     end
% end
% %===========    assign with hungarian method    ======
% [c,~] = hungarian(-G);
% newL2 = zeros(nClass,1);
% for i=1:nClass
%     newL2(L2 == i) = c(i);
% end
% end
% 
% function [B,wRA] = calRA(E, dc)
% if nargin < 2
%     dc=0.8;
% end
% 
% [n,M] = size(E); %no. of data points and no. of clusterings
% [E, no_allcl] = relabelCl(E); % re-labelling clusters in the ensemble E
% wcl = weightCl(E);
% 
% %---find pair-wise similarity of clusters in each clustering using connected triple algorithm-----
% wCT = zeros(no_allcl,no_allcl); % create matrix wCT (weighted-connected trple of clusters), pair-wise similarity matrix for each pair of clusters
% maxCl = max(E);
% minCl = min(E);
% for q = 1:M % for each clustering
%     for i=minCl(q):maxCl(q) %for each cluster
%         Ni = wcl(i,:);
%         for j=i+1:maxCl(q) %for other clusters
%             Nj = wcl(j,:);
%             wCT(i,j) = sum(min(Ni,Nj));
%         end
%     end
% end
% if max(max(wCT)) > 0
%     wCT = wCT / max(max(wCT))*dc;
% end
% wCT = wCT + wCT';
% for i = 1:no_allcl
%     wCT(i,i) = 1;
% end
% 
% cluster_num=max(max(E));
% wRA=zeros(n,cluster_num);
% B={};
% %---compute RA matrix--------------------------------
% for q = 1:M % for each clustering
%     for i=1:n
%         wRA(i, minCl(q):maxCl(q))=wCT(E(i,q), minCl(q):maxCl(q));
%         wRA(i,E(i,q))=1;
%     end
%     B{q} = wRA(:, minCl(q):maxCl(q));
% end
% end


function [U,V,ev0,ev] = svdfast(A, c)
[n,m] = size(A);
% use SVD to compute U and V
S = A'*A;
[V, ev0, ev]=eig1(S,m);
V = V(:,1:c);
U=(A*V)./(ones(n,1)*sqrt(ev0(1:c)'));
end


