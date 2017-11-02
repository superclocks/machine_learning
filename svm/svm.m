close all
clear all
load('training_matrix.txt');


%svmStruct = svmtrain(training_matrix(:,2:3),training_matrix(:,1),'showplot',true,'Method','SMO','Kernel_Function','rbf','Autoscale','false');


load fisheriris
data = [meas(:,1), meas(:,2)];
groups = ismember(species,'setosa');
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data,groups,'showplot',true,'Method','SMO','Kernel_Function','rbf','Autoscale','false');



