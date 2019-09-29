function Dataset = SMOTE(Dataset, N)
% Add n samples to Dataset by doing interpolationi
% Dataset: (attributes_num, samples_num)
% N: the number of samples that need to increase.
ori_num = size(Dataset, 2);
for i = 1:N
    sample1 = Dataset(:, unidrnd(ori_num));
    sample2 = Dataset(:, unidrnd(ori_num));
    weight = rand();
    new_sample = sample1*weight + sample2*(1-weight);
    Dataset = [Dataset, new_sample];
end
end