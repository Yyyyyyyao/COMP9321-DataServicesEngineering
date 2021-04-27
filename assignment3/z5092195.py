import sys 
import numpy as np
import pandas as pd
import sklearn
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import math
import datetime
import os
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# get the json value under the tag 'name' from the data
def get_json_names(x):
    json_data = json.loads(x)
    list_genres = []
    for token in json_data:
        list_genres.append(token['name'])
    return list_genres

# given the input training and validation data and process the data to get the training set and validation set
# flag is used to distinguish the part1 and part2 data processing.
# if flag==1:part2 data processing
# if flag==0: part1 data processing.
def format_training_data(df_training, df_validation, flag):

    # create two dataframe to store the training set and validation set
    training_dataset = pd.DataFrame(columns=['movie_id'])
    validation_dataset = pd.DataFrame(columns=['movie_id'])

    # movie_id:
    training_dataset['movie_id'] = df_training['movie_id']
    validation_dataset['movie_id'] = df_validation['movie_id']
    
    # revenue:
    training_dataset['revenue'] = df_training['revenue']
    validation_dataset['revenue'] = df_validation['revenue']
    
    # budget
    if flag == 1:
        training_dataset['budget_cat_min_30000000'] = df_training['budget'].apply(lambda x: 1 if (x <=30000000) else 0)
        training_dataset['budget_cat_30000000_40000000'] = df_training['budget'].apply(lambda x: 1 if (x >30000000)&(x<=40000000) else 0)
        training_dataset['budget_cat_40000000_50000000'] = df_training['budget'].apply(lambda x: 1 if (x >40000000)&(x<=50000000) else 0)
        training_dataset['budget_cat_50000000_60000000'] = df_training['budget'].apply(lambda x: 1 if (x >50000000)&(x<=60000000) else 0)
        training_dataset['budget_cat_60000000_70000000'] = df_training['budget'].apply(lambda x: 1 if (x >60000000)&(x<=70000000) else 0)
        training_dataset['budget_cat_70000000_80000000'] = df_training['budget'].apply(lambda x: 1 if (x >70000000)&(x<=80000000) else 0)
        training_dataset['budget_cat_80000000_90000000'] = df_training['budget'].apply(lambda x: 1 if (x >80000000)&(x<=90000000) else 0)
        training_dataset['budget_cat_90000000_100000000'] = df_training['budget'].apply(lambda x: 1 if (x >90000000)&(x<=100000000) else 0)
        training_dataset['budget_cat_100000000_130000000'] = df_training['budget'].apply(lambda x: 1 if (x >100000000)&(x<=130000000) else 0)
        training_dataset['budget_cat_130000000_160000000'] = df_training['budget'].apply(lambda x: 1 if (x >130000000)&(x<=160000000) else 0)
        training_dataset['budget_cat_160000000_190000000'] = df_training['budget'].apply(lambda x: 1 if (x >160000000)&(x<=190000000) else 0)
        training_dataset['budget_cat_190000000_max'] = df_training['budget'].apply(lambda x: 1 if (x >=190000000) else 0)
        
        validation_dataset['budget_cat_min_30000000'] = df_validation['budget'].apply(lambda x: 1 if (x <=30000000) else 0)
        validation_dataset['budget_cat_30000000_40000000'] = df_validation['budget'].apply(lambda x: 1 if (x >30000000)&(x<=40000000) else 0)
        validation_dataset['budget_cat_40000000_50000000'] = df_validation['budget'].apply(lambda x: 1 if (x >40000000)&(x<=50000000) else 0)
        validation_dataset['budget_cat_50000000_60000000'] = df_validation['budget'].apply(lambda x: 1 if (x >50000000)&(x<=60000000) else 0)
        validation_dataset['budget_cat_60000000_70000000'] = df_validation['budget'].apply(lambda x: 1 if (x >60000000)&(x<=70000000) else 0)
        validation_dataset['budget_cat_70000000_80000000'] = df_validation['budget'].apply(lambda x: 1 if (x >70000000)&(x<=80000000) else 0)
        validation_dataset['budget_cat_80000000_90000000'] = df_validation['budget'].apply(lambda x: 1 if (x >80000000)&(x<=90000000) else 0)
        validation_dataset['budget_cat_90000000_100000000'] = df_validation['budget'].apply(lambda x: 1 if (x >90000000)&(x<=100000000) else 0)
        validation_dataset['budget_cat_100000000_130000000'] = df_validation['budget'].apply(lambda x: 1 if (x >100000000)&(x<=130000000) else 0)
        validation_dataset['budget_cat_130000000_160000000'] = df_validation['budget'].apply(lambda x: 1 if (x >130000000)&(x<=160000000) else 0)
        validation_dataset['budget_cat_160000000_190000000'] = df_validation['budget'].apply(lambda x: 1 if (x >160000000)&(x<=190000000) else 0)
        validation_dataset['budget_cat_190000000_max'] = df_validation['budget'].apply(lambda x: 1 if (x >=190000000) else 0)
    
    else:
        training_dataset['budget'] = df_training['budget']
        validation_dataset['budget'] = df_validation['budget']
    
    # rating
    training_dataset['rating'] = df_training['rating']
    validation_dataset['rating'] = df_validation['rating']

    # homepage:
    training_dataset['has_homepage'] = 0
    training_dataset.loc[df_training['homepage'].isnull() == False, 'has_homepage'] = 1 #1 here means it has home page
    
    validation_dataset['has_homepage'] = 0
    validation_dataset.loc[df_validation['homepage'].isnull() == False, 'has_homepage'] = 1 #1 here means it has home page
    
    # original_language:
    unique_languages = list(df_training["original_language"].apply(pd.Series).stack().unique())
    for language in unique_languages :
        training_dataset[language] = df_training['original_language'].apply(lambda x: 1 if x == language else 0)
    for language in unique_languages :
        validation_dataset[language] = df_validation['original_language'].apply(lambda x: 1 if x == language else 0)
    
    # Genres: # may improve by looking at how correlated to revenue
#     unique_genres = list(training_dataset["genres"].apply(pd.Series).stack().unique())

    # get top 10 mean-revenue genres
#     temp = df_training[['movie_id','revenue','genres']].copy()
#     temp['genres'] = input_training['genres'].apply(lambda x: get_json_names(x))
#     unique_genres = list(temp["genres"].apply(pd.Series).stack().unique())

#     dic={}
#     for a in unique_genres:
#         mask = temp['genres'].apply(lambda x: a in x)
#         dic[a] = temp[mask]['revenue'].mean()

#     t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'genre'})
#     genres_list = list(t.sort_values(by=["mean_revenue"],ascending=False)["genre"])[:10]
    
    
    # get  10 most appearing genres
    temp = input_training[['movie_id','revenue','genres']].copy()
    temp['genres'] = temp['genres'].apply(lambda x: get_json_names(x))
    genres_list = list(temp['genres'].apply(pd.Series).stack().value_counts().index)[:10]
    
    # Apply One hot encoding to "Genres"
    training_dataset['genres'] = df_training['genres'].apply(lambda x: get_json_names(x))
    for genres in genres_list :
        training_dataset['genre_'+genres]=training_dataset['genres'].apply(lambda x: 1 if genres in x else 0)
    training_dataset = training_dataset.drop(['genres'], axis=1)

    validation_dataset['genres'] = df_validation['genres'].apply(lambda x: get_json_names(x))
    for genres in genres_list :
        validation_dataset['genre_'+genres]=validation_dataset['genres'].apply(lambda x: 1 if genres in x else 0)
    validation_dataset = validation_dataset.drop(['genres'], axis=1)

    # Production company
    # Top 10 companies who produce most movies
    temp = input_training[['movie_id','revenue','production_companies']].copy()
    temp['production_companies'] = input_training['production_companies'].apply(lambda x: get_json_names(x))
    production_company_list = list(temp['production_companies'].apply(pd.Series).stack().value_counts().index)[:10]
    
    training_dataset['production_companies'] = df_training['production_companies'].apply(lambda x: get_json_names(x))
    for company in production_company_list :
        training_dataset['company_'+company]=training_dataset['production_companies'].apply(lambda x: 1 if company in x else 0)
    training_dataset = training_dataset.drop(['production_companies'], axis=1)
    
    validation_dataset['production_companies'] = df_validation['production_companies'].apply(lambda x: get_json_names(x))
    for company in production_company_list :
        validation_dataset['company_'+company]=validation_dataset['production_companies'].apply(lambda x: 1 if company in x else 0)
    validation_dataset = validation_dataset.drop(['production_companies'], axis=1)
    
    
    # crew 
    temp = df_training[['movie_id','revenue','crew']].copy()
    temp['crew'] = temp['crew'].apply(lambda x: get_json_names(x))
    crew_list = list(temp['crew'].apply(pd.Series).stack().value_counts().index)[:10]
    
    training_dataset['crew'] = df_training['crew'].apply(lambda x: get_json_names(x))
    for crew in crew_list :
        training_dataset['crew_'+crew]=training_dataset['crew'].apply(lambda x: 1 if crew in x else 0)
    training_dataset = training_dataset.drop(['crew'], axis=1)
    
    validation_dataset['crew'] = df_validation['crew'].apply(lambda x: get_json_names(x))
    for crew in crew_list :
        validation_dataset['crew_'+crew]=validation_dataset['crew'].apply(lambda x: 1 if crew in x else 0)
    validation_dataset = validation_dataset.drop(['crew'], axis=1)
    
    # runtime
#     training_dataset['runtime'] = df_training['runtime']
#     validation_dataset['runtime'] = df_validation['runtime']
    training_dataset['runtime_cat_min_60'] = df_training['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    training_dataset['runtime_cat_61_80'] = df_training['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    training_dataset['runtime_cat_81_100'] = df_training['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    training_dataset['runtime_cat_101_120'] = df_training['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    training_dataset['runtime_cat_121_140'] = df_training['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    training_dataset['runtime_cat_141_180'] = df_training['runtime'].apply(lambda x: 1 if (x >140)&(x<=180) else 0)
    training_dataset['runtime_cat_181_max'] = df_training['runtime'].apply(lambda x: 1 if (x >=180) else 0)
    
    validation_dataset['runtime_cat_min_60'] = df_validation['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    validation_dataset['runtime_cat_61_80'] = df_validation['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    validation_dataset['runtime_cat_81_100'] = df_validation['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    validation_dataset['runtime_cat_101_120'] = df_validation['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    validation_dataset['runtime_cat_121_140'] = df_validation['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    validation_dataset['runtime_cat_141_180'] = df_validation['runtime'].apply(lambda x: 1 if (x >140)&(x<=180) else 0)
    validation_dataset['runtime_cat_181_max'] = df_validation['runtime'].apply(lambda x: 1 if (x >=180) else 0)
    
    # release date
    training_dataset['release_date'] = pd.to_datetime(df_training['release_date'])
    validation_dataset['release_date'] = pd.to_datetime(df_validation['release_date'])
    
    date_parts = ["year", "weekday", "month", "quarter"]
    for part in date_parts:
        part_col = 'release_date' + "_" + part #add prefix as  "release_date" before the columne
        training_dataset[part_col] = getattr(training_dataset['release_date'].dt, part).astype(int)
        validation_dataset[part_col] = getattr(validation_dataset['release_date'].dt, part).astype(int)
    training_dataset = training_dataset.drop(['release_date'], axis=1)
    validation_dataset = validation_dataset.drop(['release_date'], axis=1)
    
    # keyword
    # get 10 most appearing
    temp = input_training[['movie_id','revenue','keywords']].copy()
    temp['keywords'] = temp['keywords'].apply(lambda x: get_json_names(x))
    keywords_list = list(temp['keywords'].apply(pd.Series).stack().value_counts().index)[:10]
    
    training_dataset['keywords'] = df_training['keywords'].apply(lambda x: get_json_names(x))
    for keyword in keywords_list :
        training_dataset['keyword_'+keyword]=training_dataset['keywords'].apply(lambda x: 1 if keyword in x else 0)
    training_dataset = training_dataset.drop(['keywords'], axis=1)

    validation_dataset['keywords'] = df_validation['keywords'].apply(lambda x: get_json_names(x))
    for keyword in keywords_list :
        validation_dataset['keyword_'+keyword]=validation_dataset['keywords'].apply(lambda x: 1 if keyword in x else 0)
    validation_dataset = validation_dataset.drop(['keywords'], axis=1)
    
    # production_countries
    # top 10 countries who produce most movies
    if flag == 0:
        temp = input_training[['movie_id','revenue','production_countries']].copy()
        temp['production_countries'] = input_training['production_countries'].apply(lambda x: get_json_names(x))
        production_countries_list = list(temp['production_countries'].apply(pd.Series).stack().value_counts().index)[:10]
        
        training_dataset['production_countries'] = df_training['production_countries'].apply(lambda x: get_json_names(x))
        for country in production_countries_list :
            training_dataset['country_'+country]=training_dataset['production_countries'].apply(lambda x: 1 if country in x else 0)
        training_dataset = training_dataset.drop(['production_countries'], axis=1)
        
        validation_dataset['production_countries'] = df_validation['production_countries'].apply(lambda x: get_json_names(x))
        for country in production_countries_list :
            validation_dataset['country_'+country]=validation_dataset['production_countries'].apply(lambda x: 1 if country in x else 0)
        validation_dataset = validation_dataset.drop(['production_countries'], axis=1)

    
    # cast
    if (flag == 0):
        temp = df_training[['movie_id','revenue','cast']].copy()
        temp['cast'] = temp['cast'].apply(lambda x: get_json_names(x))
        cast_list = list(temp['cast'].apply(pd.Series).stack().value_counts().index)[:10]
        
        training_dataset['cast'] = df_training['cast'].apply(lambda x: get_json_names(x))
        for cast in cast_list :
            training_dataset['cast_'+cast]=training_dataset['cast'].apply(lambda x: 1 if cast in x else 0)
        training_dataset = training_dataset.drop(['cast'], axis=1)
        
        validation_dataset['cast'] = df_validation['cast'].apply(lambda x: get_json_names(x))
        for cast in cast_list :
            validation_dataset['cast_'+cast]=validation_dataset['cast'].apply(lambda x: 1 if cast in x else 0)
        validation_dataset = validation_dataset.drop(['cast'], axis=1)
    
    
    # spoken_languages
    if (flag == 1):

        temp = df_training[['movie_id','revenue','spoken_languages']].copy()
        temp['spoken_languages'] = temp['spoken_languages'].apply(lambda x: get_json_names(x))
        spoken_languages_list = list(temp['spoken_languages'].apply(pd.Series).stack().value_counts().index)[:5]
        
        training_dataset['spoken_languages'] = df_training['spoken_languages'].apply(lambda x: get_json_names(x))
        for lang in spoken_languages_list :
            training_dataset['spoken_languages_'+lang]=training_dataset['spoken_languages'].apply(lambda x: 1 if lang in x else 0)
        training_dataset = training_dataset.drop(['spoken_languages'], axis=1)
        
        validation_dataset['spoken_languages'] = df_validation['spoken_languages'].apply(lambda x: get_json_names(x))
        for lang in spoken_languages_list :
            validation_dataset['spoken_languages_'+lang]=validation_dataset['spoken_languages'].apply(lambda x: 1 if lang in x else 0)
        validation_dataset = validation_dataset.drop(['spoken_languages'], axis=1)
    

    return training_dataset, validation_dataset 


def Part1_model_training_and_prediction(training_dataset, validation_dataset):

    train_X = training_dataset.drop(['revenue', 'movie_id', 'rating'], axis=1)
    test_X = validation_dataset.drop(['revenue', 'movie_id', 'rating'], axis=1)

    # log and square
    train_X['budget'] = train_X['budget'].apply(lambda x: (math.log(x))**2)
    train_y = training_dataset['revenue'].apply(lambda x: (math.log(x))**2)

    test_X['budget'] = test_X['budget'].apply(lambda x: (math.log(x))**2)
    test_y = validation_dataset['revenue'].apply(lambda x: (math.log(x))**2)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': .01, 'loss': 'ls'} 
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(train_X,train_y)
    predictions2 = clf.predict(test_X).reshape(-1,1)

    # #linear regression
    # lm = LinearRegression() #our 6th model
    # lm.fit(train_X, train_y)
    # lm_preds = lm.predict(test_X)

    # RF_model = RandomForestRegressor(random_state =0, n_estimators=500, max_depth=50)
    # RF_model.fit(train_X, train_y)
    # y_hat = RF_model.predict(test_X)
    
    res = pd.DataFrame(columns=['movie_id', 'preds', 'groundTruth'])
    res['movie_id'] = validation_dataset['movie_id']
    res['groundTruth'] = np.exp(np.sqrt(test_y))
    i = 0
    while(i < len(predictions2)):
        res.loc[i, 'preds'] = np.exp(np.sqrt(predictions2[i][0]))
        i += 1

    xmean = np.mean(res['preds'])
    ymean = np.mean(res['groundTruth'])
    res['x_minus_xmean'] = res['preds'].apply(lambda x: x-xmean)
    res['y_minus_ymean'] = res['groundTruth'].apply(lambda x: x-ymean)

    # According to the pearsons formula 
    # Calculate pearsons coefficient
    c_upper = sum(res['x_minus_xmean'] * res['y_minus_ymean'])
    c_lower = math.sqrt(sum(res['x_minus_xmean']*res['x_minus_xmean'])*sum(res['y_minus_ymean']*res['y_minus_ymean']))
    pearsons_coeffcient = c_upper / c_lower
    MSE = mean_squared_error(res['preds'], res['groundTruth'])

    summary = pd.DataFrame(columns=['zid', 'MSE', 'correlation'], data=[['z5092195', round(MSE, 2), round(pearsons_coeffcient, 2)]])

    output = pd.DataFrame(columns=['movie_id', 'predicted_revenue'])
    output['movie_id'] = res['movie_id']
    output['predicted_revenue'] = res['preds']

    return summary, output



def Part2_model_training_and_prediction(training_dataset, validation_dataset):
    
    train_X = training_dataset.drop(['revenue', 'movie_id', 'rating'], axis=1)
    test_X = validation_dataset.drop(['revenue', 'movie_id', 'rating'], axis=1)

    train_y = training_dataset['rating']
    test_y = validation_dataset['rating']

    # create the model
    clf = LinearDiscriminantAnalysis().fit(train_X, train_y)
    preds = clf.predict(test_X)

    avg_precision, avg_recall, _, _ = precision_recall_fscore_support(test_y, preds, average='macro')
    accuracy = accuracy_score(test_y, preds)
    summary = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'], data=[['z5092195', round(avg_precision,2), round(avg_recall,2), round(accuracy,2)]])

    output = pd.DataFrame(columns=['movie_id', 'predicted_rating'])
    output['movie_id'] = validation_dataset['movie_id']
    output['predicted_rating'] = preds

    return summary, output

if __name__ == '__main__':

    # Read arguments
    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]

    input_training = pd.read_csv(train_csv)
    input_validation = pd.read_csv(validate_csv)

    # Deal with possible missing data
    input_training['budget'] = input_training['budget'].fillna(input_training['budget'].mean())
    input_validation['budget'] = input_validation['budget'].fillna(input_validation['budget'].mean())
    input_training['runtime'] = input_training['runtime'].fillna(input_training['runtime'].mean())
    input_validation['runtime'] = input_validation['runtime'].fillna(input_validation['runtime'].mean())
    input_training['rating'] = input_training['rating'].fillna(2)
    input_validation['rating'] = input_validation['rating'].fillna(2)
    input_training['revenue'] = input_training['revenue'].fillna(input_training['revenue'].mean())
    input_validation['revenue'] = input_validation['revenue'].fillna(input_validation['revenue'].mean())
    
    training_dataset, validation_dataset = format_training_data(input_training, input_validation, flag=0)
    part1_summary, part1_output = Part1_model_training_and_prediction(training_dataset, validation_dataset)
    part1_summary.to_csv('z5092195.PART1.summary.csv', index=False)
    part1_output = part1_output.sort_values(by='movie_id')
    part1_output.to_csv('z5092195.PART1.output.csv', index=False)

    training_dataset, validation_dataset = format_training_data(input_training, input_validation, flag=1)
    part2_summary, part2_output = Part2_model_training_and_prediction(training_dataset, validation_dataset)
    part2_summary.to_csv('z5092195.PART2.summary.csv', index=False)
    part2_output = part2_output.sort_values(by='movie_id')
    part2_output.to_csv('z5092195.PART2.output.csv', index=False)





















