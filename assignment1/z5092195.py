import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math

studentid = os.path.basename(sys.modules[__name__].__file__)


def deg2rad(deg):
    return deg * (math.pi/180)


def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    
    R = 6373 #Radius of the earth in km
    dLat = deg2rad(lat2-lat1)  #deg2rad below
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c#Distance in km

    return d



def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df_exposure = pd.read_csv(exposure, sep=';')
    df_countries = pd.read_csv(countries)
    df_exposure.loc[7, 'country'] = 'Cape Verde'
    df_exposure.loc[13, 'country'] = 'Palestinian Territory'
    df_exposure.loc[28, 'country'] = 'United States'
    df_exposure.loc[45, 'country'] = 'Democratic Republic of the Congo'
    df_exposure.loc[56, 'country'] = 'North Korea'
    df_exposure.loc[73, 'country'] = 'Laos'
    df_exposure.loc[88, 'country'] = 'Republic of the Congo'
    df_exposure.loc[97, 'country'] = 'Brunei'
    df_exposure.loc[108, 'country'] = 'Vietnam'
    df_exposure.loc[138, 'country'] = 'Swaziland'
    df_exposure.loc[150, 'country'] = 'Ivory Coast'
    df_exposure.loc[155, 'country'] = 'Macedonia'
    df_exposure.loc[159, 'country'] = 'Moldova'
    df_exposure.loc[161, 'country'] = 'Russia'
    df_exposure.loc[179, 'country'] = 'South Korea'

    df1 = pd.merge(left=df_exposure, right=df_countries, on=None, left_on='country', right_on='Country')
    df1.drop(['country'], axis=1, inplace=True)
    df1.set_index(['Country'],inplace=True)
    df1.sort_index(inplace=True)
    #################################################
    
    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df2 = df1.copy()
    df2["avg_latitude"] = np.nan
    df2["avg_longitude"] = np.nan
    i = 0
    while i < df2.shape[0]:
        
        latitude_list=[]
        longitude_list=[]
        json_string = df2.iloc[i]['Cities'].split("|||")
        for token in json_string:
            a_json = json.loads(token)
            latitude_list.append(a_json["Latitude"])
            longitude_list.append(a_json["Longitude"])
        mean_latitude = np.mean(latitude_list)
        mean_longitude = np.mean(longitude_list)
        df2["avg_latitude"][i] = mean_latitude
        df2["avg_longitude"][i] = mean_longitude
        i+=1
    #################################################
    

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df3 = df2.copy()
    df3["distance_to_Wuhan"] = np.nan
    for index, row in df3.iterrows():
        df3.loc[index, "distance_to_Wuhan"] = getDistanceFromLatLonInKm(30.5928, 114.3055, row["avg_latitude"], row["avg_longitude"])
    df3.sort_values("distance_to_Wuhan", inplace=True)
    #################################################
    
    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df4=df2.copy()
    df_continets=pd.read_csv("Countries-Continents.csv")
    df4.reset_index(level=0, inplace=True)
    df4.loc[25, 'Country'] = 'Burkina'
    df4.loc[41, 'Country'] = 'CZ'
    df4.loc[42, 'Country'] = 'Congo'
    df4.loc[114, 'Country'] = 'Burma (Myanmar)'
    df4.loc[123, 'Country'] = 'Korea, North'
    df4.loc[128, 'Country'] = 'East Timor'
    df4.loc[137, 'Country'] = 'Congo'
    df4.loc[139, 'Country'] = 'Russian Federation'
    df4.loc[157, 'Country'] = 'Korea, South'
    df4.loc[180, 'Country'] = 'US'
    df4.set_index(['Country'],inplace=True)
    df4.sort_index(inplace=True)
    df_merged = pd.merge(left=df4, right=df_continets, on=None, left_index=True, right_on='Country')
    selected_columns = df_merged[["Continent", "Covid_19_Economic_exposure_index"]]
    df_result=selected_columns.copy()
    # df_result.replace('x', '0', regex=True, inplace=True)
    df_result.replace('x', np.nan, regex=True, inplace=True)
    df_result.replace(',', '.', regex=True, inplace=True)
    df_result["Covid_19_Economic_exposure_index"] = pd.to_numeric(df_result["Covid_19_Economic_exposure_index"])
    df_final = df_result.groupby(["Continent"])['Covid_19_Economic_exposure_index'].agg('mean').reset_index()
    df4 = df_final.set_index(['Continent'])
    df4.sort_values("Covid_19_Economic_exposure_index",inplace=True)
    df4.rename(columns={"Covid_19_Economic_exposure_index": "average_covid_19_Economic_exposure_index"}, inplace=True)
    #################################################

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df5=df2.copy()
    selected_columns = df5[["Income classification according to WB", "Foreign direct investment", "Net_ODA_received_perc_of_GNI"]]
    df5_target=selected_columns.copy()
    df5_target.replace('x', np.nan, regex=True, inplace=True)
    df5_target.replace('No data', np.nan, regex=True, inplace=True)
    df5_target.dropna()
    df5_target.replace(',', '.', regex=True, inplace=True)
    df5_target["Foreign direct investment"] = pd.to_numeric(df5_target["Foreign direct investment"])
    df5_target["Net_ODA_received_perc_of_GNI"] = pd.to_numeric(df5_target["Net_ODA_received_perc_of_GNI"])
    df5_target.rename(columns={"Income classification according to WB": "Income Class", "Foreign direct investment":"Avg Foreign direct investment", "Net_ODA_received_perc_of_GNI": "Avg_Net_ODA_received_perc_of_GNI"}, inplace=True)

    df5 = df5_target.groupby(["Income Class"])['Avg Foreign direct investment', 'Avg_Net_ODA_received_perc_of_GNI'].agg('mean')
    #################################################

    

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []
    #################################################
    # Your code goes here ...
    selected_columns = df2[["Cities", "Income classification according to WB"]]
    df_target=selected_columns.copy()
    df_LIC = df_target[df_target["Income classification according to WB"]=='LIC']
    df_LIC.reset_index(inplace=True)
    df_info = df_LIC.copy()
    df_info["City"] = np.nan
    df_info["Population"] = np.nan

    i = 0
    while i< df_info.shape[0]:
        city_list = []
        population_list = []
        json_string = df_info.iloc[i]['Cities'].split("|||")
        for token in json_string:
            a_json = json.loads(token)
            if a_json['Population'] != None:
                city_list.append(a_json['City'])
                population_list.append(str(a_json['Population']))

        df_info["City"][i] = city_list
        df_info["Population"][i] = population_list
        i +=1
    df_info.drop(columns=['Cities', 'Income classification according to WB'], inplace=True)
    df6 = df_info.set_index(['Country']).apply(pd.Series.explode).reset_index()
    df6["Population"] = pd.to_numeric(df6["Population"])
    df6.sort_values(by=['Population'], ascending=False, inplace=True)
    cities_lst = df6.head(5)['City'].values.tolist()
    #################################################
    


    log("QUESTION 6", output_df=None, other=cities_lst)
    return cities_lst


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    selected_columns = df2[["Cities"]]
    df_target=selected_columns.copy()
    df_target.reset_index(inplace=True)
    df_info = df_target.copy()
    df_info["City"] = np.nan

    i = 0
    while i< df_info.shape[0]:
        city_list = []
        json_string = df_info.iloc[i]['Cities'].split("|||")
        for token in json_string:
            a_json = json.loads(token)
            city_list.append(a_json['City'])

        df_info["City"][i] = city_list
        i +=1
    df_info.drop(["Cities"], axis=1, inplace=True)
    df_process = df_info.set_index(['Country']).apply(pd.Series.explode).reset_index()
    df_non_dup = df_process.drop_duplicates()
    df_findDup = df_non_dup[df_non_dup.groupby('City')['Country'].transform('count')>1]
    df7 = df_findDup.groupby('City')['Country'].apply(lambda x: list(x)).reset_index()
    df7.rename(columns={'City':'city', 'Country':'coutries'}, inplace=True)
    df7.set_index(['city'], inplace=True)
    #################################################

    

    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    selected_columns = df2[["Cities"]]
    df_cities=selected_columns.copy()
    df_cities.reset_index(inplace=True)
    df_info = df_cities.copy()
    df_info["Population"] = np.nan
    i = 0
    while i< df_info.shape[0]:
        city_list = []
        population_list = []
        json_string = df_info.iloc[i]['Cities'].split("|||")
        for token in json_string:
            a_json = json.loads(token)
            if a_json['Population'] != None:
                population_list.append(str(a_json['Population']))
        df_info["Population"][i] = population_list
        i +=1
    df_info.drop(columns=["Cities"], axis=1, inplace=True)
    df6 = df_info.set_index(['Country']).apply(pd.Series.explode).reset_index()
    df6["Population"] = pd.to_numeric(df6["Population"])
    
    df_country_population = df6.groupby(["Country"])["Population"].agg('sum').reset_index()
    df_country_population.loc[25, 'Country'] = 'Burkina'
    df_country_population.loc[41, 'Country'] = 'CZ'
    df_country_population.loc[42, 'Country'] = 'Congo'
    df_country_population.loc[114, 'Country'] = 'Burma (Myanmar)'
    df_country_population.loc[123, 'Country'] = 'Korea, North'
    df_country_population.loc[128, 'Country'] = 'East Timor'
    df_country_population.loc[137, 'Country'] = 'Congo'
    df_country_population.loc[139, 'Country'] = 'Russian Federation'
    df_country_population.loc[157, 'Country'] = 'Korea, South'
    df_country_population.loc[180, 'Country'] = 'US'
    df_continets=pd.read_csv(continents)
    df_merged = pd.merge(left=df_country_population, right=df_continets, on=None, left_on='Country', right_on='Country')
    world_total_population = df_merged["Population"].sum()
    df_south_america = df_merged[df_merged["Continent"]=="South America"]
    df_south_america["population_percentage"] = df_south_america["Population"].apply(lambda x: x/world_total_population)
    df_plot = df_south_america[["Country", "population_percentage"]]
    df_plot.set_index("Country", inplace=True)

    df_res = df_plot.copy()
    df_res["Population percentage in the world"] = df_res["population_percentage"].apply(lambda x: round(x*100, 2))
    df_res.drop("population_percentage", axis=1, inplace=True)
    
    ax = df_res.plot(kind='bar', title ="Percentage of the world population living in South American country", figsize=(15, 10), legend=True, fontsize=12)
    #################################################

    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    # df2.reset_index(inplace=True)
    # selected_columns = ["Income classification according to WB", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import" , "Foreign direct investment, net inflows percent of GDP", "Foreign direct investment"]
    # df_target = df2[selected_columns]
    # df_target.replace('x', np.nan, regex=True, inplace=True)
    # df_target.replace(',', '.', regex=True, inplace=True)

    # df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI"] = pd.to_numeric(df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI"])
    # df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import"] = pd.to_numeric(df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import"])
    # df_target["Foreign direct investment"] = pd.to_numeric(df_target["Foreign direct investment"])
    # df_target["Foreign direct investment, net inflows percent of GDP"] = pd.to_numeric(df_target["Foreign direct investment, net inflows percent of GDP"])
    # df9 = df_target.groupby(["Income classification according to WB"])["Income classification according to WB", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import" , "Foreign direct investment, net inflows percent of GDP", "Foreign direct investment"].agg('mean')
    # ax = df9.plot(kind='bar', title ="Comparison between Different income class", figsize=(15, 10), legend=True, fontsize=12)

    df2.reset_index(inplace=True)
    selected_columns = ["Income classification according to WB", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import" , "Foreign direct investment, net inflows percent of GDP", "Foreign direct investment"]
    df_target = df2[selected_columns]
    df_target.replace('x', np.nan, regex=True, inplace=True)
    df_target.replace(',', '.', regex=True, inplace=True)

    df_target["Avg Covid_19_Economic_exposure_index_Ex_aid_and_FDI"] = pd.to_numeric(df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI"])
    df_target["Avg Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import"] = pd.to_numeric(df_target["Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import"])
    df_target["Avg Foreign direct investment"] = pd.to_numeric(df_target["Foreign direct investment"])
    df_target["Avg Foreign direct investment, net inflows percent of GDP"] = pd.to_numeric(df_target["Foreign direct investment, net inflows percent of GDP"])
    df9 = df_target.groupby(["Income classification according to WB"])["Income classification according to WB", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI", "Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import" , "Foreign direct investment, net inflows percent of GDP", "Foreign direct investment"].agg('mean')
    df9.sort_index()
    # ax = df9.plot(kind='bar', title ="Comparison between Different income class", figsize=(15, 10), legend=True, fontsize=12)


    order = ['LIC', 'MIC', 'HIC']
    mapping = {income_class: i for i, income_class in enumerate(order)}
    key = df9.index.map(mapping)
    df_res = df9.iloc[key.argsort()]
    ax = df_res.plot(kind='bar', title ="Comparison between Different income class", figsize=(15, 10), legend=True, fontsize=12)
    ax.grid()
    #################################################
    
    plt.savefig("{}-Q12.png".format(studentid))

def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    #################################################
    # Your code goes here ...
    df_continents = pd.read_csv(continents)
    df2.reset_index(inplace=True)
    df2.loc[25, 'Country'] = 'Burkina'
    df2.loc[41, 'Country'] = 'CZ'
    df2.loc[42, 'Country'] = 'Congo'
    df2.loc[114, 'Country'] = 'Burma (Myanmar)'
    df2.loc[123, 'Country'] = 'Korea, North'
    df2.loc[128, 'Country'] = 'East Timor'
    df2.loc[137, 'Country'] = 'Congo'
    df2.loc[139, 'Country'] = 'Russian Federation'
    df2.loc[157, 'Country'] = 'Korea, South'
    df2.loc[180, 'Country'] = 'US'
    df_merged = pd.merge(left=df2, right=df_continents, on=None, left_on='Country', right_on='Country')
    selected_columns=['Country', 'avg_latitude','avg_longitude', 'Cities', 'Continent']
    df_target = df_merged[selected_columns]
    df_info = df_target.copy()
    df_info["Population"] = np.nan

    i = 0
    while i< df_info.shape[0]:
        population_list = []
        json_string = df_info.iloc[i]['Cities'].split("|||")
        for token in json_string:
            a_json = json.loads(token)
            if a_json['Population'] != None:
                population_list.append(a_json['Population'])

        df_info["Population"][i] = sum(population_list)
        i +=1
    df_info.drop(["Cities"], axis=1, inplace=True)

    a, b = 50, 500
    x, y = df_info["Population"].min(), df_info["Population"].max()
    df_info['p_norm'] = (df_info["Population"] - x) / (y - x) * (b - a) + a
    s = [n for n in list(df_info["p_norm"])]

    fig, ax = plt.subplots()
    colors = {'Asia': 'red', 'Europe':'blue', 'Africa':'black', 'North America':'yellow', 'South America':'orange','Oceania':'purple'}

    grouped = df_info.groupby('Continent')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='avg_longitude', y='avg_latitude', label=key, color=colors[key], s='p_norm', figsize=(15, 10))
    #################################################
    
    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")