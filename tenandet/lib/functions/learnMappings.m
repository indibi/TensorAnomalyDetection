function [B, U, objAll] = learnMappings(Y, X, modes)
%learnMappings Learn Mappings between tensor Y and labels X
% Inputs:
%   [B, U] = learnMappings(Y, X, modes)
%   Y     : Tensor to be mapped.
%   X     : Binary tensor of labels.
%   modes : modes along which mappings will be learnt.
% Outputs:
%   B     : The bias tensor
%   U     : Learnt tensor factors.

tensor_size  = size(Y);
N            = tensor_size(end);
M            = length(tensor_size);
imodes = setdiff(1:M, modes);
m            = length(modes);
nLoop        = 1;
beta         = 10^0;
zeta         = .1;
gamma_1      = 10^0;
gamma_2      = 10^0;
ranks        = 20*ones(1, 2*m-1);%2:length(modes)*2;
%ranks(3)     = 25;

F  = zeros(tensor_size);
S1 = zeros(tensor_size);
S2 = zeros(tensor_size);
B  = zeros(tensor_size(1:M-1));
Umodes = [modes, 2*M-modes(end:-1:1)+1]; tensor_size = size(Y);
U  = uInit([tensor_size, tensor_size(end:-1:1)], Umodes, ranks);
Yp = mergeTensors(Y, U(1:m), modes);
Yp = ipermute(squeeze(mergeFactors({Yp, U{m+1:end}})),  [setdiff(1:M, modes), modes(end:-1:1)]);

iter    = 1;
dIter   = inf;
max_iter = 10^2+1;
min_dist = 10^-2;
objAll = [];
after_D = [];
after_F = [];
after_Y = [];
while iter<=max_iter && dIter >= min_dist
    %% D Update
    % $||D||_1 + \frac{\beta}{2}||D||_F^2 + \frac{\gamma_1}{2}||D-X+\sigma(F)-S1||_F^2$
%     dTime   = cputime;
    D       = soft_threshold((X - lreLU(F, zeta) + S1).*gamma_1./(beta+gamma_1), 1/(beta+gamma_1));
    after_D = [after_D, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
%     dTime   = cputime - dTime
    
    %% F Update
    % $\frac{\gamma_1}{2}||D-X+\sigma(F)-S1||_F^2 +
    % \frac{\gamma_2}{2}||F-Yp-B-S2||_F^2$
%     fTime   = cputime;
    ind     = F >= 0;
    temp1   = X - D + S1;
    temp2   = Yp + B + S2;
    F(ind)  = (gamma_1*temp1(ind)+gamma_2*temp2(ind))/(gamma_1+gamma_2);
    F(~ind) = (zeta*gamma_1*temp1(~ind)+gamma_2*temp2(~ind))/(zeta*gamma_1+gamma_2);%Yp(~ind)+S2(~ind);
    after_F = [after_F, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
%     fTime   = cputime-fTime
    %% U update
    if mod(iter, nLoop)==0
        % $||F-YxU-B-S2||_F^2$
%         [Utemp, val] = updateU(F-B-S2, Y, U, modes);
% %         after_U = [after_U, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
%         Yp      = mergeTensors(Y, Utemp(1:m), modes);
%         Yp      = ipermute(squeeze(mergeFactors({Yp, Utemp{m+1:end}})), [setdiff(1:M, modes), modes(end:-1:1)]);
        Utemp = update_Umat(Y, F-B-S2, modes, ranks(3));
        Yp = reshape(Utemp*t2m(Y, modes), [tensor_size(modes), tensor_size(imodes)]);
        Yp = ipermute(Yp, [modes, imodes]);
        after_Y = [after_Y, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
        if after_Y(end)>after_F(end)
            disp('Error in U update.')
        end
        U = Utemp;
    end
   
    %% B Update
    % $||F-YxU-B-S2||_F^2$
    B       = mean(F - Yp - S2, M);
%     after_B = [after_B, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
    objAll  = [objAll, obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)];
   
    %% S1 Update
    S1      = S1 - D + X - lreLU(F, zeta);
    
    %% S2 Update
    S2      = S2 - F + Yp + repmat(B, [ones(1, M-1), N]);
    
%     sum(abs(D),'all')
    dIter   = norm(ten2vec(sign(F)-X))/norm(X(:));
    if dIter<min_dist
        disp('Hit the error threshold.')
    end
    if iter>=max_iter
        disp('Hit max iteration.')
    end
    iter    = iter+1;
end

end

function [val, terms] = obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)
% [val, terms] = obj_func(D,X,F,Yp,B,S1,S2,beta,gamma_1,gamma_2,zeta)
% Objective value calculator.
terms(1) = sum(abs(D),'all') + beta/2*norm(D(:))^2;
terms(2) = gamma_1/2*sum((D-X+lreLU(F, zeta)-S1).^2,'all');
terms(3) = gamma_2/2*sum((F-Yp-B-S2).^2,'all');
val = sum(terms);
end

% function [D, U] = learnMappings(Y, X, modes)
% %learnMappings Learn Mappings between tensor Y and labels X
% %   [D, U] = learnMappings(Y, X, modes)
% %   Y     : Tensor to be mapped.
% %   X     : Binary tensor of labels.
% %   modes : modes along which mappings will be learnt.
% 
% tensor_size  = size(Y);
% nLoop        = 20;
% beta         = 10^0;
% zeta         = .2;
% gamma_1      = 10^0;
% gamma_2      = 10^0;
% ranks        = 5*ones(1, 2*length(modes)-1);%2:length(modes)*2;
% %ranks(3)     = 25;
% 
% F  = zeros(tensor_size);
% S1 = zeros(tensor_size);
% S2 = zeros(tensor_size);
% D  = F;
% Umodes = [modes, 2*ndims(Y)-length(modes)+modes]; tensor_size = size(X);
% U  = uInit([size(Y), tensor_size(end:-1:1)], Umodes, ranks);
% Yp = mergeTensors(Y, U(1:length(modes)), modes);
% Yp = permute(squeeze(mergeFactors({Yp, U{length(modes)+1:end}})), ndims(Y):-1:1);
% 
% 
% iter    = 1;
% dIter   = inf;
% max_iter = 2*10^3+1;
% min_dist = 10^-6;
% while iter<=max_iter && dIter >= min_dist
% %     dTime   = cputime;
%     Dtemp   = D;
%     D       = softThresh((X - lreLU(F, zeta) + S1)*gamma_1/(beta+gamma_1), 1/(beta+gamma_1));
% %     dTime   = cputime - dTime
%     
% %     fTime   = cputime;
%     ind     = F>=0;
%     F(ind)  = (gamma_1*(X(ind)+S1(ind)-D(ind))+gamma_2*(Yp(ind)+S2(ind)))/(gamma_1+gamma_2);
%     F(~ind) = (zeta*gamma_1*(X(~ind)+S1(~ind)-D(~ind))+gamma_2*(Yp(~ind)+S2(~ind)))/(zeta*gamma_1+gamma_2);%Yp(~ind)+S2(~ind);
% %     fTime   = cputime-fTime
%     
%     if mod(iter, nLoop)==1
%         U       = updateU(F, Y, S2, U, modes);
%         Yp      = mergeTensors(Y, U(1:length(modes)), modes);
%         Yp      = permute(squeeze(mergeFactors({Yp, U{length(modes)+1:end}})), ndims(Y):-1:1);
%     end
%     
%     S1      = S1 - D - lreLU(F, zeta) + X;
%     
%     S2      = S2 - F + Yp;
%     
%     %objAll  = sum(abs(D),'all') + beta/2*norm(D(:))^2 + ...
%      %   gamma_1/2*sum((D-X+reLU(F)-S1).^2,'all') + gamma_2/2*sum((F-Yp-S2).^2,'all');
%     %sum(abs(D),'all');
%     Dtemp   = lreLU(F, zeta)-X;
%     dIter   = norm(Dtemp(:));
%     if dIter<min_dist
%         disp('Hit the error threshold.')
%     end
%     iter    = iter+1;
% end
% 
% end