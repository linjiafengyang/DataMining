load('D:\linjiafengyang\Code\Python\PrincipalComponentAnalysis\yale_face.mat');
k = 5; % ��ά��ά��
% ��ֵͼ��
meanFace = mean(X, 2);
% imshow(reshape(meanFace, [64 64]), []);
X_mean_norm = X - meanFace;
sigma = cov(X_mean_norm'); % ����Э����
tic;
[V_eig,D] = eig(sigma); % V_eigΪ������������DΪ����ֵ
time_eig = toc;
D = diag(D); % ��Ϊ�ԽǾ���
V_eig = (rot90(V_eig))'; % ��������������Ӵ�С����
D = rot90(rot90(D)); % ������ֵ����Ӵ�С����
V_eig_reduce = V_eig(:,1:k); % ǰ�����������
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