function [S,multiply] = Compute_S(X,A,W,P,multiply)
m = size(A,2);
n = size(X,2);
lambda = 1;
eta = 1;
% viewNum = size(X,2);
% for i = 1 : viewNum
%     multiply{i}(1:n) = 0;
% end
%     for i = 1 : viewNum
        M = sqdist(X,A); 
        H = lambda * eye(m) + eta * W*W';
        G = 2 * eta * W*P' - M';  
        T = G' * pinv(H);
        R = pinv(H);
        a(1:m) = 1;
        r = R*a';
        
        for j = 1 : n
            S(j,:) = 1/2 * (T(j,:)- multiply(j) * r');
        end
        
        for number = 1 : n
            temp = (T(number,:)-multiply(1,number)*r(:,1)');
%             multiply{i}(1,number) = 1/2*(sum(find(a>0)));
            multiply_new(1,number) = sum(temp(find(temp>0)));
%             if multiply
%             for column = 1:m
%                 multiply{i}(1,number) = 1/2 *(T{i}(number,column) - multiply{i}(1,number) * r(column, 1)) - 1;    
%             end
        end
        multiply_new = 1/2 * multiply_new - 1;
        fi = sum(r(:,1));
        multiply = multiply - multiply_new/fi;

            
%     end
end

