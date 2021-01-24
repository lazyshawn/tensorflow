% 将每次实验的数据制作成子数据集
clear; clc;

%% => 遍历所有仿真数据文件夹
list = dir('../cylinder/P*');
n = length(list);
for i = 1:n
    folder = list(i).name;
    mkDataSet(folder);
end

%% => 将一次仿真数据保存为子数据集
function mkDataSet( folder )
    path = '../cylinder/';
    f_pos = [path, folder, '/position.txt'];
    f_vel = [path, folder, '/velocity.txt'];

    data_pos = load(f_pos);
    data_vel = load(f_vel);

    % => 数据分类
    time = data_pos(:,1);
    x_1 = data_pos(:,2);
    x_2 = data_pos(:,3);
    z_1 = data_pos(:,4);
    z_2 = data_pos(:,5);

    u_1 = data_vel(:,2);
    u_2 = data_vel(:,3);
    w_1 = data_vel(:,4);
    w_2 = data_vel(:,5);

    % => 数据处理
    % Twist
    X = (x_1 + x_2)/2;
    Z = (z_1 + z_2)/2;
    Theta = atan2(z_2-z_1, x_2-x_1);
    d2 = X.*X + Z.*Z;
    % Velocity
    U = (u_1 + u_2)/2;
    W = (w_1 + w_2)/2;
    Omega = (((u_1-u_2).*(u_1-u_2)./(4*d2)) + ((w_1-w_2).*(w_1-w_2)./(4*d2))).^(1/2);

    % Record
    twist = [time, X, Z, Theta];  % character
    velocity = [time, U, W, Omega];  % label
    subDataSet = [twist(1:end-20,:), velocity(1:end-20,2:end), velocity(2:end-19,2:end)];

    % => 保存结果
    save(['./data/', folder,'_subSet.txt'], 'subDataSet', '-ascii');
end
