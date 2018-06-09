load('D:\linjiafengyang\Code\Python\PrincipalComponentAnalysis\yale_face.mat');
k = 5; % ��ά��ά��
% ��ֵͼ��
meanFace = mean(X, 2);
% imshow(reshape(meanFace, [64 64]), []);
X_mean_norm = X - meanFace;
sigma = cov(X_mean_norm'); % ����Э����
tic;
% svd����
[U,S,V] = svd(X_mean_norm);
% ǰ�����������
Ureduce = U(:,1:k);
time_svd = toc;
for i = 1:k
    subplot(1,k,i);
    imshow(reshape(Ureduce(:,i), [64 64]), []);
end
z = Ureduce' * X_mean_norm;
X_approx = Ureduce * z;
ratio = 0;
S_diag = diag(S);
for i = 1:k
    r = S_diag(i)/sum(S_diag);
    ratio = ratio + r;
%     if (ratio >= 0.9)
%         break;
%     end
end

