import json

import pandas as pd
import flask
from flask import Flask
from flask import request
from flask import send_file
from flask_restx import Resource, Api
from flask_restx import fields
from flask_restx import inputs
from flask_restx import reqparse
import requests

import sqlite3
from sqlite3 import Error

import re
from datetime import datetime, timedelta

import math

import matplotlib
import matplotlib.pyplot as plt

from flask import jsonify, make_response

app = Flask(__name__)
api = Api(app,
          default="tv-shows",  # Default namespace
          title="tv shows database",  # Documentation Title
          description="This is an API to get and return a tv show")  # Documentation Description

# a nested model in show_model
country_model = api.model('Country', {
    'name': fields.String,
    'code': fields.String,
    'timezone': fields.String
})

# a nested model in show_model
schedule_model = api.model('Schedule', {
    'time': fields.String,
    'days': fields.List(fields.String)
})

# a nested model in show_model
rating_model = api.model('Rating', {
    'average': fields.Integer,
})

# a nested model in show_model
network_model = api.model('Network', {
    'id': fields.Integer,
    'name': fields.String,
    'country': fields.Nested(country_model)
})

# The following is the schema of Book
show_model = api.model('Show', {
    'tvmaze-id': fields.Integer,
    'name': fields.String,
    'type': fields.String,
    'language': fields.String,
    'genres': fields.List(fields.String),
    'status': fields.String,
    'runtime': fields.Integer,
    'premiered': fields.String,
    'officialSite': fields.Url,
    "schedule": fields.Nested(schedule_model),
    "rating": fields.Nested(rating_model),
    "weight": fields.Integer,
    "network": fields.Nested(network_model),
    "summary": fields.String
})


# define params
parser = reqparse.RequestParser()
parser.add_argument('name')
parser.add_argument('page', type=int, default=1)
parser.add_argument('page_size', type=int, default=100)
parser.add_argument('filter', action='split',  default=['id','name'])
parser.add_argument('order_by', action='split', default=['+id'])
parser.add_argument('format', choices=['json', 'image'])
parser.add_argument('by', choices=['language', 'genres', 'status', 'type'])

# Q6
@api.route('/tv-shows/statistics')
@api.param('format', 'Choose what image/json to display the statistics')
@api.param('by', 'Choose language/genres/status/type to see the statistics')
class Tvshows_stats(Resource):
    @api.response(200, 'OK')
    @api.response(404, 'No available TV shows to display')
    @api.response(400, 'Error')
    @api.doc(description="Get the stats of an attribute of the TV shows in a chosen format")
    def get(self):
        # build db connection
        conn = create_connection(r"z5092195.db")

        # receive the query parameters
        args = parser.parse_args()
        format = args.get('format')
        by = args.get('by')

        # Check if there are TV shows to display
        sql_check_exist = ''' SELECT * FROM tvshows; '''
        data = retrive_data(conn, sql_check_exist)
        if len(data) == 0: # No Tv shows found
            return {"message": "There is not TV show found in db"}, 404
        else:
            df_rows = [] # list used to create a dataframe to process data
            count_num_updates = 0
            for row in data:
                db_id = row[0]
                show_info = json.loads(row[1])
                last_update_date = row[2]
                show_tvmaze_id = row[3]
                df_single_row = []
                df_single_row = [db_id, last_update_date, show_info[by]]
                df_rows.append(df_single_row)

                # check the last 24 hours tv shows
                now = datetime.now()
                old_time = now-timedelta(hours=24)
                old_time = old_time.strftime("%Y-%m-%d-%H:%M:%S")
                now_time = now.strftime("%Y-%m-%d-%H:%M:%S")
                if old_time <= last_update_date <= now_time:
                    count_num_updates += 1
            # create a dataframe
            df = pd.DataFrame(df_rows, columns=['id', 'last_update_date', by])
        df_group = df.copy()
        if by == 'genres': # genres may have multiple values, need to use explode
            df_explode = df.set_index(['id', 'last_update_date'])
            df_explode = df_explode.explode(by)
            df_group = df_explode.reset_index()
        value_dic = {} # generate the statistics for the content
        df_group = df_group.groupby([by]).agg('count')
        for index, row in df_group.iterrows():
            value_dic[index] = row[('id')]
        sum_values = sum(value_dic.values()) # sum_values to do the percentage calculation
        plot_dic = {} # create a dictionary for future ploting
        plot_dic['by'] = [] # one key is the "by" attribute
        plot_dic['percentage'] = [] # another key is percentage
        for key, value in value_dic.items():
            value_dic[key] = round((value / sum_values)*100,1)
            plot_dic['by'].append(key)
            plot_dic['percentage'].append(round((value / sum_values)*100,1))

        # create the plot data frame
        df_plot = pd.DataFrame(data=plot_dic).set_index('by')
        if format == 'json':
            result = {
                'total': len(data),
                'total-update': count_num_updates,
                'value': value_dic
            }
            return result, 200
        else:
            # if format is an image
            matplotlib.use('agg')
            fig = df_plot.plot(kind='pie',  figsize=(20, 16), fontsize=12, y='percentage', autopct=lambda x: str(int(x))+"%", title="Total {} TV shows statistics by '{}' which has {} updated in the last 24 hours".format(len(data), by, count_num_updates)).get_figure()
            fig.savefig('test.png')
            filename = 'test.png'
            return send_file(filename, mimetype='image/png', cache_timeout=0)

# Q5
@api.route('/tv-shows')
@api.param('filter', 'Choose what attribtues to display')
@api.param('page_size', 'Number of items in a page')
@api.param('page', 'Choose which page to display')
@api.param('order_by', 'Page items are order by these attributes')
class Tvshows_retrieve(Resource):
    @api.response(200, 'OK')
    @api.response(404, 'No available TV shows to display')
    @api.response(400, 'Error')
    @api.doc(description="Retrieve the list of available Tv shows")
    def get(self):
        conn = create_connection(r"z5092195.db")

        # receive the query parameters
        args = parser.parse_args()
        order_by = args.get('order_by')
        page = args.get('page')
        page_size = args.get('page_size')
        filter = args.get('filter')

        if page_size == 0:
            return {"message": "Invalid page size"}, 400

        # check order by validation
        order_by_choices = ['id','name','runtime','premiered','rating-average']
        for val in order_by:
            print(val)
            if val[0] not in ['+', '-']:
                return {"message": "Invalid order by signs, should be '+' or '-'"}, 400
            if val[1:] not in order_by_choices:
                return {"message": "Invalid order by attribute"}, 400
        
        # check filter validation
        filter_choices = ['tvmaze_id' ,'id' ,'last-update' ,'name' ,'type' ,'language' ,'genres' ,'status' ,'runtime' ,'premiered' ,'officialSite' ,'schedule' ,'rating' ,'weight' ,'network' ,'summary']
        for val in filter:
            if val not in filter_choices:
                return {"message": "Invalid filter attribute"}, 400

        sql_count_rows = '''SELECT COUNT(*) FROM tvshows'''
        num_of_rows_in_db = retrive_data(conn, sql_count_rows)[0][0]

        tolerate_page_num = math.ceil(num_of_rows_in_db / page_size)
        if page > tolerate_page_num:
            return {"message": "Invalid page number"}, 400
        
        conn = create_connection(r"z5092195.db")
        # check if this tvshow is already in the db
        sql_check_exist = ''' SELECT * FROM tvshows; '''
        data = retrive_data(conn, sql_check_exist)
        if len(data) == 0:
            return {"message": "There is not TV show found in db"}, 404
        else:
            df_rows = []
            for row in data:
                db_id = row[0]
                show_info = json.loads(row[1])
                last_update_date = row[2]
                show_tvmaze_id = row[3]
                df_single_row = []
                for filter_val in filter:
                    if filter_val == 'tvmaze_id':
                        df_single_row.append(show_tvmaze_id)
                    elif filter_val == 'id':
                        df_single_row.append(db_id)
                    elif filter_val == 'last-update':
                        df_single_row.append(last_update_date)
                    else:
                        df_single_row.append(show_info[filter_val])

                df_append_order_columns = []
                for order_val in order_by:
                    order_content = order_val[1:]
                    if order_content == 'rating-average':
                        df_single_row.append(show_info['rating']['average'])
                        df_append_order_columns.append('rating-average')
                    elif order_content in filter:
                        continue
                    else:
                        df_append_order_columns.append(order_content)
                        df_single_row.append(show_info[order_content])

                df_rows.append(df_single_row)

            df = pd.DataFrame(df_rows, columns=filter+df_append_order_columns)
            sign_to_sort = []
            values_to_sort = []
            for order_val in order_by:
                sign = order_val[0]
                content = order_val[1:]
                values_to_sort.append(content)
                if sign == '+':
                    sign_to_sort.append(True)
                else:
                    sign_to_sort.append(False)

            df_result = df.sort_values(values_to_sort, ascending=sign_to_sort)
            df_result = df_result.reset_index(drop=True)

        
        start_item_index = (page-1)*page_size
        if num_of_rows_in_db <= page*page_size:
            end_item_index = num_of_rows_in_db-1
        else:
            end_item_index = page*page_size-1
        
        tv_shows_list = []
        for index in range(start_item_index, end_item_index+1):
            single_tv_show = {}
            for val in filter:
                single_tv_show[val] = str(df_result.iloc[index][val])
            tv_shows_list.append(single_tv_show)


        order_by_string = ','.join(order_by)
        filter_string = ','.join(filter)
        self_link = "http://"+flask.request.host+"/tv-shows?order_by="+order_by_string+"&page="+str(page)+"&page_size="+str(page_size)+"&filter="+filter_string
        next_link = "http://"+flask.request.host+"/tv-shows?order_by="+order_by_string+"&page="+str(page+1)+"&page_size="+str(page_size)+"&filter="+filter_string
        prev_link = "http://"+flask.request.host+"/tv-shows?order_by="+order_by_string+"&page="+str(page-1)+"&page_size="+str(page_size)+"&filter="+filter_string
        if page == 1:
            if page*page_size >= num_of_rows_in_db:
                links = {
                    "self":{
                        "href": self_link
                    }
                }
            else:
                
                links = {
                    "self":{
                        "href": self_link
                    },
                    "next":{
                        "href": next_link
                    }
                }
        else:
            if page*page_size >= num_of_rows_in_db:
                links = {
                    "self":{
                        "href": self_link
                    },
                    "previous":{
                        "href": prev_link
                    }
                }
            else:
                links = {
                    "self":{
                        "href": self_link
                    },
                    "previous":{
                        "href": prev_link
                    },
                    "next":{
                        "href": next_link
                    }
                }


        result = {
            "page": page,
            "page-size": page_size,
            "tv-shows": tv_shows_list,
            "_links": links
        }
    
        return result
        
# Q2-Q4
@api.route('/tv-shows/<int:id>')
@api.param('id', 'The show identifier')
class Tvshows_operation(Resource):

    # Q2
    @api.response(404, 'tvshow was not found')
    @api.response(200, 'OK')
    @api.doc(description="Get a tvshow by its ID")
    def get(self, id):
        # build databse connection
        conn = create_connection(r"z5092195.db")
        # check if this tvshow is already in the db
        sql_check_exist = ''' SELECT * FROM tvshows
                                WHERE id={}; '''.format(id)
        data = retrive_data(conn, sql_check_exist)
        if len(data) == 0: # Tv show does not found in db
            return {"message": "Tvshow with id {} is not found in the tvmaze.com".format(id)}, 404
        else:
            # if exist, extract the data from db
            row = data[0]
            show_info = json.loads(row[1])
            last_update_date = row[2]
            show_tvmaze_id = row[3]
            
            # get the prev and next id
            sql_get_next_id = ''' SELECT id FROM tvshows
                                    WHERE id>{} LIMIT 1; '''.format(id)
            sql_get_prev_id = ''' SELECT id FROM tvshows
                                    WHERE id<{} 
                                    ORDER BY id DESC
                                    LIMIT 1; '''.format(id)

            # create links if exists
            self_link = "http://"+flask.request.host+"/tv-shows/"+str(id)
            next_id_row = retrive_data(conn, sql_get_next_id)
            prev_id_row = retrive_data(conn, sql_get_prev_id)
            if len(next_id_row) != 0: # if it has next tv show in df
                next_id = next_id_row[0][0] # get the next tv show id
                if len(prev_id_row) != 0: # if it has prev tv show in df
                    prev_id = prev_id_row[0][0] # get the next tv show id

                    # put them into the links dic
                    next_link = "http://"+flask.request.host+"/tv-shows/"+str(next_id)
                    prev_link = "http://"+flask.request.host+"/tv-shows/"+str(prev_id)
                    links = {"self": {"href": self_link},
                                "previous": {"href": prev_link},
                                "next": {"href": next_link}
                            }
                else: # if it does not have prev dic link 
                    # put them into the links dic
                    next_link = "http://"+flask.request.host+"/tv-shows/"+str(next_id)
                    links = {"self": {"href": self_link},
                                "next": {"href": next_link}
                            }
            else: # same procedure to non-next tvshow
                if len(prev_id_row) != 0:
                    prev_id = prev_id_row[0][0]
                    prev_link = "http://"+flask.request.host+"/tv-shows/"+str(prev_id)
                    links = {"self": {"href": self_link},
                                "previous": {"href": prev_link}
                            }
                else:
                    links = {"self": {"href": self_link}
                            }
            
            # get infos about the tv-show
            result_dic = {"tvmaze-id" :show_tvmaze_id,
                            "id": id,
                            "last-update": last_update_date,
                            "name": show_info['name'],
                            "type": show_info['type'],
                            "language": show_info['language'],
                            "genres": show_info['genres'],
                            "status": show_info['status'],
                            "runtime": show_info['runtime'],
                            "premiered": show_info['premiered'],
                            "officialSite": show_info['officialSite'],
                            "schedule": show_info['schedule'],
                            "rating": show_info['rating'],
                            "weight": show_info['weight'],
                            "network": show_info['network'],
                            "summary": show_info['summary'] ,
                            "_links": links
            }
            return result_dic, 200

    # Q3
    @api.response(404, 'tvshow was not found')
    @api.response(200, 'OK')
    @api.doc(description="Delete a tvshow by its ID")
    def delete(self, id):
        conn = create_connection(r"z5092195.db")
        # check if this tvshow is already in the db
        sql_check_exist = ''' SELECT * FROM tvshows
                                WHERE id={}; '''.format(id)
        data = retrive_data(conn, sql_check_exist)
        if len(data) == 0: # if no such Tv show in db, return error msg
            return {"message": "Tvshow with id {} is not found in the tvmaze.com".format(id)}, 404
        else:
            # delete the show
            # using sql command
            sql_delete_command = '''DELETE FROM tvshows
                                    WHERE id={};'''.format(id)
            delete_data(conn, sql_delete_command)
            return {"message": "The tv show with id {} was removed from the database!".format(id),
                    "id": id}, 200


    # Q4
    @api.response(404, 'tvshow was not found')
    @api.response(200, 'OK')
    @api.doc(description="Update a tvshow by its ID")
    @api.expect(show_model, validate=True)
    def patch(self, id):
        show = request.json
        conn = create_connection(r"z5092195.db")
        # check if this tvshow is already in the db
        sql_check_exist = ''' SELECT * FROM tvshows
                                WHERE id={}; '''.format(id)
        data = retrive_data(conn, sql_check_exist)
        if len(data) == 0: # if the tv is not found
            return {"message": "Tvshow with id {} is not found in the tvmaze.com".format(id)}, 404
        else:
            # if exist, extract the data from db
            row = data[0]
            show_info = json.loads(row[1])

            # get newest update time
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d-%H:%M:%S")
            new_last_update_date = date_time

            # Due the structure design of db
            # if user changes tvmaze-id, i need to update that in db
            if 'tvmaze-id' in show.keys():
                new_show_tvmaze_id = show['tvmaze-id']
            else:
                new_show_tvmaze_id = row[3]
            
            # change the content json in db
            for key, value in show.items():
                show_info[key] = value

            new_show_info_string = json.dumps(show_info)
            update_record = (new_show_info_string, new_last_update_date, new_show_tvmaze_id, id)
            # update into database
            update_data(conn, update_record)

            # return the result
            self_link = "http://"+flask.request.host+"/tv-shows/"+str(id)
            links = {"self": {"href": self_link}}
            result = {"id": id, 
                        "last-update": new_last_update_date,
                        "_links": links
            }
            return result, 200


# Q1
@api.route('/tv-shows/import')
@api.param('name', 'the name of the show')
class Tvshows(Resource):

    @api.response(201, 'Created')
    @api.response(404, 'Tv show not found')
    @api.response(400, 'Create Failed')
    @api.doc(description="Import shows from tvmaze")
    def post(self):
        # establish database connection
        conn = create_connection(r"z5092195.db")
        # retrieve the query parameters
        args = parser.parse_args()
        show_name = args.get('name')

        # real request
        show_url = 'http://api.tvmaze.com/search/shows?q='+show_name
        show_raw = requests.get(show_url).json()

        
        if len(show_raw) == 0: # check whether it get a tvshow from the tvmaze.com
            return {"message": "Tvshow {} is not found in the tvmaze.com".format(show_name)}, 404
        else:
            # Compare whether the name matches
            # Good Girls == gOOd^girlS
            # Good Girls1 != Good Girls
            name_from_tvmaze = show_raw[0]['show']['name']
            name_from_tvmaze = re.sub('[^0-9a-zA-Z]+', '', name_from_tvmaze)
            name_from_param = show_name
            name_from_param = re.sub('[^0-9a-zA-Z]+', '', name_from_param)
            if name_from_param.lower() == name_from_tvmaze.lower():
                # take the first match
                show_info = show_raw[0]
                # check if this tvshow is already in the db
                sql_check_duplicates = ''' SELECT * FROM tvshows
                                            WHERE tvmaze_id={}; '''.format(show_info['show']['id'])
                data = retrive_data(conn, sql_check_duplicates)
                if len(data) == 0: # it did not find the dupicates
                    # get import time 
                    # used as last-update time 
                    now = datetime.now()
                    date_time = now.strftime("%Y-%m-%d-%H:%M:%S")

                    # dumps the whole tvshow into a string
                    show_info_string = json.dumps(show_info['show'])

                    # form db record
                    show_record = (show_info_string, date_time, show_info['show']['id'])
                    # insert into database
                    db_id = insert_data(conn, show_record)
                    self_link = "http://"+flask.request.host+"/tv-shows/"+str(db_id)
                    links = {"self": {"href": self_link}}
                    result = {"id": db_id, 
                                "last-update": date_time,
                                "tvmaze-id": show_info['show']['id'],
                                "_links": links
                    }
                    return result, 201
                else: # Found dulicates in db
                    return {"message": "Tvshow {} is already exist in the db".format(show_name)}, 400
            else:
                return {"message": "Tvshow {} is not found in the tvmaze.com".format(show_name)}, 404

        return {"message": "Unknow error occurs"}, 400


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

def create_table(conn):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    # create table in the db
    sql_create_tvshows_table = """ CREATE TABLE IF NOT EXISTS tvshows (
                                        id integer PRIMARY KEY,
                                        show_info text,
                                        last_update text,
                                        tvmaze_id integer
                                    ); """

    try:
        c = conn.cursor()
        c.execute(sql_create_tvshows_table)
    except Error as e:
        print(e)

def insert_data(conn, tvshow):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO tvshows(show_info,last_update, tvmaze_id)
                VALUES(?,?,?) '''
    c = conn.cursor()
    c.execute(sql, tvshow)
    conn.commit()

    return c.lastrowid

def retrive_data(conn, sql_command):
    cur = conn.cursor()
    cur.execute(sql_command)
    rows = cur.fetchall()
    return rows

def delete_data(conn, sql_command):
    cur = conn.cursor()
    cur.execute(sql_command)
    conn.commit()

def update_data(conn, record):
    sql_update_data = ''' UPDATE tvshows
                                    SET show_info=?,
                                        last_update=?,
                                        tvmaze_id=?
                                    WHERE id=?; '''
    cur = conn.cursor()
    cur.execute(sql_update_data, record)
    conn.commit()

if __name__ == "__main__":
    # initialize connection to the db
    conn = create_connection(r"z5092195.db")
    if conn is not None:
        # create table 
        create_table(conn)
        # data_1 = ('Cool App with SQLite & Python', '2015-01-01', '2015-01-30')
        # id = insert_data(conn, data_1)
    else:
        print("Error: Cannot create the database connection")
 
    app.run(debug=True)