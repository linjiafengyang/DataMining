load('D:\linjiafengyang\Code\Python\PrincipalComponentAnalysis\yale_face.mat');
k = 5; % 降维的维数
% 均值图像
meanFace = mean(X, 2);
% imshow(reshape(meanFace, [64 64]), []);
X_mean_norm = X - meanFace;
sigma = cov(X_mean_norm'); % 计算协方差
tic;
[V_eig,D] = eig(sigma); % V_eig为右特征向量，D为特征值
time_eig = toc;
D = diag(D); % 化为对角矩阵
V_eig = (rot90(V_eig))'; % 将特征向量矩阵从大到小排序
D = rot90(rot90(D)); % 将特征值矩阵从大到小排序
V_eig_reduce = V_eig(:,1:k); % 前五个特征向量
for i = 1:k
    subplot(1,k,i);
    imshow(reshape(V_eig_reduce(:,i), [64 64]), []);
end
Z_eig = V_eig_reduce' * X_mean_norm;
X_approx = V_eig_reduce * Z_eig;
ratio = 0;
for i = 1:k
    r = D(i)/sum(D);
    ratio = ratio + r;
%     if (ratio >= 0.9)
%         break;
%     end
end