function [L] = apply_lof(Y, K)

L = zeros(size(Y));
for i=1:size(Y,1)
    for j=1:size(Y,2)
        for k=1:size(Y,4)
        [~,L(i,j,:,k)] = LOF(squeeze(Y(i,j,:,k)),K);
        end
    end
end

end