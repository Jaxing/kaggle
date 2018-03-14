# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
import pandas
# import csv

training_data = pandas.read_csv('./train.tsv', delimiter='\t')
test_data = pandas.read_csv('./test.tsv', delimiter='\t')
sample = pandas.read_csv('./sample_submission.csv')

training_data = training_data
item_condition_id = (training_data.loc[:, ['item_condition_id', 'price']])\
    .groupby('item_condition_id')\
    .agg({'price': [np.max, np.mean, np.min]})
category_name = (training_data.loc[:, ['category_name', 'price']])#['category_name'].apply(lambda x: pandas.Series(str(x).split('/')))
training_data['category_1'] = training_data.category_name.str.extract('([^/]+)/[^/]+/[^/]+', expand=False)
training_data['category_2'] = training_data.category_name.str.extract('[^/]+/([^/]+)/[^/]+', expand=False)
training_data['category_3'] = training_data.category_name.str.extract('[^/]+/[^/]+/([^/]+)', expand=False)
category_name1 = training_data.loc[:, ['category_1', 'price']]\
    .groupby('category_1')\
    .agg({'price': [np.max, np.mean, np.min]})
category_name2 = training_data.loc[:, ['category_2', 'price']]\
    .groupby('category_2')\
    .agg({'price': [np.max, np.mean, np.min]})
category_name3 = training_data.loc[:, ['category_3', 'price']]\
    .groupby('category_3')\
    .agg({'price': [np.max, np.mean, np.min]})

brand_name = training_data.loc[:, ['brand_name', 'price']]
#brand_name['brand_name'] = (brand_name['brand_name'].apply(lambda x: "brand" if not pandas.isnull(x) else "no brand"))
brand_name = brand_name\
    .groupby('brand_name')\
    .agg({'price': [np.max, np.mean, np.min]})


print("Item conditions")
print(item_condition_id)
print(item_condition_id.describe())

print("Brand names")
print(brand_name)
print(brand_name.describe())

print("Categories 1")
print(category_name1)
print(category_name1.describe())

print("Categories 2")
print(category_name2)
print(category_name2.describe())

print("Categories 3")
print(category_name3)
print(category_name3.describe())
# print(training_data.count(axis=item_condition_id))
# print(training_data.count(axis=brand_name))

print(training_data.loc[:,
      ["brand_name", "category_1"]\
      .group('brand_name')\
      .arg({"category_1": []})
      )