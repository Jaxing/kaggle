# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import pandas
# import csv

training_data = pandas.read_csv('./train.tsv', delimiter='\t')
test_data = pandas.read_csv('./test.tsv', delimiter='\t')
sample = pandas.read_csv('./sample_submission.csv')

# training_data_input = (training_data.loc[:, ['Pclass', 'Sex']])\
#                             .replace('female', 1)\
#                             .replace('male', -1)
# training_data_target = training_data.loc[:, 'Survived']
#
#
# test_data_input = (test_data.loc[:, ['Pclass', 'Sex']])\
#                         .replace('female', 1)\
#                         .replace('male', -1)
training_data = training_data
item_condition_id = (training_data.loc[:, ['item_condition_id', 'price']])\
    .groupby('item_condition_id')\
    .agg({'price': [np.max, np.mean, np.min]})
category_name = (training_data.loc[:, ['category_name', 'price']])['category_name'].apply(lambda x: pandas.Series(str(x).split('/')))
category_name = category_name\
    .groupby('category_name')\
    .agg({'price': [np.max, np.mean, np.min]})

brand_name = training_data.loc[:, ['brand_name', 'price']]
brand_name['brand_name'] = (brand_name['brand_name'].apply(lambda x: "brand" if not pandas.isnull(x) else "no brand"))
brand_name = brand_name\
    .groupby('brand_name')\
    .agg({'price': [np.max, np.mean, np.min]})


print("Item conditions")
print(item_condition_id)
print(item_condition_id.describe())

print("Brand names")
print(brand_name)
print(brand_name.describe())

print("Categories")
print(category_name)
print(category_name.describe())
# print(training_data.count(axis=item_condition_id))
# print(training_data.count(axis=brand_name))

