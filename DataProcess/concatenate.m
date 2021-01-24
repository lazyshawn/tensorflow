% 整合数据集
clear;  clc;

%% => 读取数据
directory = './data/';
list = dir([directory, '*.txt']);
n = length(list);

%% => 测试集数据
rem = floor(rand(1)*n)+1;
data = load([directory, list(rem).name]);
save('./dataSet/test_data.txt', 'data', '-ascii');
list(rem) = [];

%% => 训练集、验证集数据
dataSet = [];
for i = 1:n-1
    file = list(i).name;
    data = load([directory, file]);
    dataSet = [dataSet; data];
end

save('./dataSet/dataSet.txt', 'dataSet', '-ascii');
