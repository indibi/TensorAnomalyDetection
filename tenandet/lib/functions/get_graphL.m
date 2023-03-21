function [L, W] = get_graphL(X, K, varargin)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
n = ndims(X);
S = size(X,n);

if length(varargin)==2
    C = length(unique(varargin{2}));
end

D = zeros(S,S);
if length(varargin)==1
    X = reshape(X,[],S);
    for s=1:S
        for sp=1:S
            D(s,sp) = norm(X(:,s)-X(:,sp))^2;
        end
    end
    ind_knn = knnsearch(X', X', 'K', K+1);
    ind_knn = ind_knn(:,2:end);
    map = zeros(S);
    for s=1:S
        map(s, ind_knn(s, :)) = 1;
    end
    map = map | map';
    W = zeros(S);
    const = floor(log10(sum(var(D))));
    gamma = 1/(S*10^const);
    W(map) = exp(-D(map)./gamma);
%     dist_mat = norm(X(:))*ones(S);
%     data = reshape(X, [], S);
%     load neighbors.mat
%     for i=1:S
%         curr_zone = data(:,i);
%         if varargin{1}
%             [~,neighbor_ind ] = intersect(regions, neighbors{i});
%         else
%             neighbor_ind = 1:S;
%         end
%         dist_mat(i, neighbor_ind) = sum((curr_zone-data(:,neighbor_ind)).^2,1)./(norm(curr_zone)*sqrt(sum(data(:,neighbor_ind).^2,1)));
%     end
%     W = exp(-(triu(dist_mat)+triu(dist_mat,1)'));
%     W = W-eye(size(W));
%     vec = sort(W(:));
%     vec(vec==inf)=[];
%     W(W<.4*vec(end))=0;
%     W = W+eye(size(W));
% %     W = W>.4*vec(end);
else
    X = reshape(X,[],S);
    W = D;
    
    for s=1:S
        for sp=1:S
            D(s,sp) = norm(X(:,s)-X(:,sp))^2;
        end
    end
    if length(varargin)==3
        D = exp(-D/varargin{3}) + 10^-8;
        W = D+D';
        W = W./kron(ones(1, S),sum(W,2));
    else
        D=D+diag(sum(D));
        if length(K)==1
            sdW = repmat(max(mink(D,K)),S,1);
            D(D>sdW) = 0;
            W = D;
            W = W | W';
        elseif  length(K)==2
            for c=1:C
                ind        = labels==c;
                ceilD      = repmat(max(mink(D(ind,ind), K(1))),sum(ind),1);
                W(ind,ind) = D(ind,ind) <= ceilD;
                floorD     = repmat(max(mink(D(~ind, ind), K(2))),S-sum(ind),1);
                W(~ind,ind)= D(~ind,ind)<= floorD;
            end
            W = W | W';
            W = W.*kron((eye(C)-2\ones(C))*2,ones(S/C));
        end
    end
end
D = diag(sum(W));
L = D-W;
% W = normc(W);
end