RawData = load('training_data.txt');

Training = RawData(randperm(fix(size(RawData,1)*0.0)),:);
Validation = RawData(randperm(fix(size(RawData,1)*0.0)),:);
Testing = RawData(randperm(fix(size(RawData,1)*1.0)),:);

csvwrite('raw_data_.csv', RawData);
%csvwrite('training_data_.csv', Training);
%csvwrite('validation_data_.csv', Validation);
csvwrite('testing_data_.csv', Testing);