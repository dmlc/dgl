#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:13:00 2019

@author: Rajiv Sambasivan
"""

import pandas as pd
from arango import ArangoClient
import time
import traceback
#import simplejson
import uuid
import networkx as nx
import dgl
#import numpy as np
from collections import OrderedDict

import torch as th
import yaml
import os
import requests
import json
from custom_http_client import CustomHTTPClient
from amlnx_adapter.dgl.dgl_networkx_arango_adapter import DGL_Networkx_Arango_Adapter

class ITSM_Dataloader:

    def __init__(self, input_file = "pp_recoded_incident_event_log.csv",\
                 create_db = True, frac = 0.10):
        self.emlg = None
        self.db = None
        self.labels = list()
        self.vertex_list = None
        self.edge_dict = {}
        self.feature_dict = {}
        self.feature_data = None
        self.setup_schema()
        self.sampling_frac = frac
        self.replication_factor = None
        self.cfg = None
        self.batch_vert_dict = None
        self.batch_edge_dict = None
        
        if create_db:
            self.input_file = input_file
            self.delete_db()
            self.create_db()
            self.create_graph()
            
            self.load_data()
        else:
            self.set_db_connection()

        return


    def setup_schema(self):
        self.vertex_list = ['incident', 'support_org', 'customer', 'vendor']

        self.edge_dict = {'incident-support_org': {'from': 'incident', 'to': 'support_org'},\
                          'incident-customer': {'from': 'incident', 'to': 'customer'},\
                          'incident-vendor': {'from': 'incident', 'to': 'vendor'}}
        

        self.feature_dict['support_org'] = ['assignment_group', 'assigned_to']
        self.feature_dict['customer'] = ['opened_by']
        self.feature_dict['vendor'] = ['vendor']
        self.feature_data = {v : OrderedDict() for v in self.vertex_list}

        self.feature_dict['incident'] = ['D_sys_mod_count', 'D_reopen_count',\
'urgency','incident_state', 'u_symptom', 'impact', 'contact_type',\
                          'u_priority_confirmation', 'cmdb_ci', 'rfc',  'problem_id',\
                          'caused_by', 'location', 'knowledge', 'resolved_by', 'subcategory',\
                          'active', 'category', 'priority', 'reassigned']

        return


    def set_db_connection(self):
        self.cfg = self.get_conn_config()
        db_conn_protocol = self.cfg['arangodb']['conn_protocol']
        db_srv_host = self.cfg['arangodb']['DB_service_host']
        db_srv_port = self.cfg['arangodb']['DB_service_port']

        
        
        host_connection = db_conn_protocol + "://" + db_srv_host + ":" + str(
            db_srv_port)
        ms_user_name = self.cfg['arangodb']['username']
        ms_password =  self.cfg['arangodb']['password']
        ms_dbName = self.cfg['arangodb']['dbName']
        
        client = ArangoClient(hosts= host_connection,\
                              http_client=CustomHTTPClient(username = ms_user_name,\
                                                           password = ms_password))
        

        db = client.db(ms_dbName, ms_user_name, ms_password)

        self.db = db
        self.emlg = self.db.graph('ITSMg')

    def delete_db(self):

        client = ArangoClient(hosts= 'http://localhost:8529')

        sys_db = client.db('_system',\
                       username='root',
                       password='open sesame')


        if sys_db.has_database('ITSM_db'):
            sys_db.delete_database('ITSM_db')

        return
    
    def get_conn_config(self, file_path = 'arango_ms_conn.yaml'):
        file_name = os.path.join(os.path.dirname(__file__), file_path)
        with open(file_name, "r") as file_descriptor:
            cfg = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        return cfg
    
    def dump_data(self):
        file_name = os.path.join(os.path.dirname(__file__),
                                 "arango_ms_conn.yaml")
        with open(file_name, "w") as file_descriptor:
            cfg = yaml.dump(self.cfg, file_descriptor)
        return cfg
    

        
    def create_db(self):
#        client = ArangoClient(hosts= 'http://localhost:8529')
#
#        # Connect to "_system" database as root user.
#        # This returns an API wrapper for "_system" database.
#        sys_db = client.db('_system',\
#                       username='root',
#                       password='open sesame')
#
#        # Create a new arangopipe database if it does not exist.
#        if not sys_db.has_database('ITSM_db'):
#            sys_db.create_database('ITSM_db')
#
#        if not sys_db.has_user('ITSM_db_admin'):
#            sys_db.create_user(username = 'ITSM_db_admin',\
#                               password = 'open sesame')
#
#        sys_db.update_permission(username = 'ITSM_db_admin',\
#                                 database = 'ITSM_db', permission = "rw")
#
#        # Connect to arangopipe database as administrative user.
#         #This returns an API wrapper for "test" database.
#        db = client.db('ITSM_db',\
#                       username='ITSM_db_admin',\
#                       password='open sesame')
#        self.db = db
        
        self.cfg = self.get_conn_config()
        db_conn_protocol = self.cfg['arangodb']['conn_protocol']
        db_srv_host = self.cfg['arangodb']['DB_service_host']
        db_srv_port = self.cfg['arangodb']['DB_service_port']
        db_end_point = self.cfg['arangodb']['DB_end_point']
        db_serv_name = self.cfg['arangodb']['DB_service_name']
        self.replication_factor = self.cfg['arangodb']['arangodb_replication_factor']
        
        
        host_connection = db_conn_protocol + "://" + db_srv_host + ":" + str(
            db_srv_port)
        print("Host Connection: " + str(host_connection))
        
        client = ArangoClient(hosts=host_connection)
            
        api_data = {}
        
        API_ENDPOINT = host_connection + "/_db/_system/" + db_end_point + \
                            "/" + db_serv_name

        r = requests.post(url=API_ENDPOINT, json=api_data, verify=False)
        
        if r.status_code == 409 or r.status_code == 400:
            print("It appears that you are attempting to connecting using \
                             existing connection information. So either set reconnect = True when you create ArangoPipeAdmin or recreate a connection config and try again!"
                )
            return

        assert r.status_code == 200, \
            "Managed DB endpoint is unavailable !, reason: " + r.reason + " err code: " +\
            str(r.status_code)
        result = json.loads(r.text)
        print("Managed service database was created !")
        ms_dbName = result['dbName']
        ms_user_name = result['username']
        ms_password = result['password']
        self.cfg['arangodb']['username'] = ms_user_name
        self.cfg['arangodb']['password'] = ms_password
        self.cfg['arangodb']['dbName'] = ms_dbName
        self.dump_data()
           
            
        

        client = ArangoClient(hosts= host_connection,\
                              http_client=CustomHTTPClient(username = ms_user_name,\
                                                           password = ms_password))
        

        db = client.db(ms_dbName, ms_user_name, ms_password)
        self.db = db
        
        return


    
    def create_graph(self):
        if not self.db.has_graph('ITSMg'):
            self.emlg = self.db.create_graph('ITSMg')
        else:
            self.emlg = self.db.graph("ITSMg")


        self.create_graph_vertices()
        self.create_graph_edges()
        return

    def create_graph_edges(self):

        for edge in self.edge_dict:
            src_vertex = self.edge_dict[edge]['from']
            dest_vertex = self.edge_dict[edge]['to']
            if not self.emlg.has_edge_definition(edge):
                self.db.create_collection(edge, edge = True,\
                                          replication_factor = self.replication_factor)
                self.emlg.create_edge_definition(edge_collection = edge,\
                                                      from_vertex_collections=[src_vertex],\
                                                      to_vertex_collections=[dest_vertex] )

        return

    def create_graph_vertices(self):
        for v in self.vertex_list:
            if not self.emlg.has_vertex_collection(v):
                self.db.create_collection(v, self.replication_factor)
                self.emlg.create_vertex_collection(v)
        return

    def trim_node_data(self):
        np_node_data = {v : pd.DataFrame() for v in self.vertex_list}
        t0 = time.time()
        print("Creating Pandas data frames...")
        for vertex in self.vertex_list:

            for row, (node_id, node_data) in enumerate(self.feature_data[vertex].items()):
                np_node_data[vertex] = np_node_data[vertex].append(node_data,\
                            ignore_index = True)


            print("Removing columns that are not needed...")
            cols = np_node_data[vertex].columns.tolist()
            if vertex == 'incident':
                cols.remove('reassigned')
                cols.remove('node_id')
            else:
                cols.remove('node_id')
            np_node_data[vertex] = np_node_data[vertex][cols]
            print("Converting to integer properties...")
            np_node_data[vertex] = np_node_data[vertex].astype(int)
            print("Done!")


        t1 = time.time()
        et = (t1 -t0)/60
        et = round(et,2)

        print ("Execution took :" + str(et) + " minutes!")


        return np_node_data

    def id_sequence(self, vertex):
        id_dict = {v: 0 for v in self.vertex_list}
        while True:
            yield id_dict[vertex]
        id_dict[vertex] += 1


    def do_inserts(self):
        for v in self.vertex_list:
            batch_docs = self.batch_vert_dict[v]
            self.db.collection(v).insert_many(batch_docs)
            self.batch_vert_dict[v] = list()
        edge_names = [*self.edge_dict]
        
        for ename in edge_names:
            batch_edge_docs = self.batch_edge_dict[ename]
            self.db.collection(ename).insert_many(batch_edge_docs)
            self.batch_edge_dict[ename] = list()
        
        return
        
    def get_networkx_graph(self, load_from_db = True, graph_desc_file_loc = "graph_descriptor.yaml"):
        dgnx = DGL_Networkx_Arango_Adapter(load_from_db = load_from_db, graph_desc_fp = graph_desc_file_loc)
        l,g = dgnx.create_networkx_graph()
        
        return l, g
    
    def load_data(self):
        t0 = time.time()
        df = pd.read_csv(self.input_file)
        df = df.sample(frac = self.sampling_frac)
        num_rows = df.shape[0] - 1
        print("A dataset with %d rows is being used for this run" % (num_rows) )
        df = df.reset_index()

        node_val_ids = {v: dict() for v in self.vertex_list}
        vertex_colls = {v: self.emlg.vertex_collection(v) for v in self.vertex_list}
        edge_names = [*self.edge_dict]
        edge_colls = {ename: self.emlg.edge_collection(ename) for ename in edge_names}
        row_vertex_map = {'incident': 'number', 'support_org': 'assignment_group',\
                          'customer': 'opened_by', 'vendor': 'vendor'}
        batch_size = 500
        self.batch_vert_dict = {v : list() for v in self.vertex_list}
        self.batch_edge_dict = {ename: list() for ename in edge_names}
        batch_count = 0
        for row_index, row in df.iterrows():
            try:
                if row_index % 500 == 0:
                    print("Processing row: " + str(row_index))
                # insert the vertices
                record_vertex_keys = dict()
                
                for v in self.vertex_list:

                    the_vertex = dict()
                    row_val = row[row_vertex_map[v]]
                    #if not row_val in node_val_ids[v]:
                    the_vertex['node_id'] = str(uuid.uuid4().int >> 64)
                    the_vertex['_key'] = v.upper() + "-" + the_vertex['node_id']
                    node_val_ids[v][row_val] = the_vertex['_key']

                    self.load_vertex_attributes(row, the_vertex, v )
                    self.batch_vert_dict[v].append(the_vertex)
                    record_vertex_keys[v] = node_val_ids[v][row_val]
            

                #insert the edges
            
                for ename in edge_names:

                    from_vertex = self.edge_dict[ename]['from']
                    to_vertex = self.edge_dict[ename]['to']
                    edge_key = record_vertex_keys[from_vertex] + "-" + \
                    record_vertex_keys[to_vertex]
                    the_edge = {"_key" : edge_key,\
                                "_from": from_vertex + "/" + record_vertex_keys[from_vertex],\
                                "_to": to_vertex + "/" + record_vertex_keys[to_vertex]}
                    self.batch_edge_dict[ename].append(the_edge)
                    
                if row_index > 0 and (row_index % batch_size == 0):
                    self.do_inserts()
                    
                
                if num_rows % batch_size != 0:
                    if row_index == num_rows:
                        self.do_inserts()
                    
                    

            except Exception as e:
                traceback.print_exc()
                

            #breakpoint()


        t1 = time.time()
        et = float((t1 -t0) / 60)
        et = round(et, 2)
        print("Data load took " + str(et) + " minutes!.")
        print("Done loading data!")

        return

    def load_vertex_attributes(self, row, the_vertex, vertex_name):

        if vertex_name == 'incident':
            self.load_incident_attributes(row, the_vertex)
        if vertex_name == 'customer':
            self.load_customer_attributes(row, the_vertex)
        if vertex_name == 'support_org':
            self.load_support_org_attributes(row, the_vertex)
        if vertex_name == 'vendor':
            self.load_vendor_attributes(row, the_vertex)

        return

    def load_incident_attributes(self, row, the_vertex):
        subset_dict = row[self.feature_dict['incident']].to_dict()


        for a in subset_dict:
            the_vertex[a] = subset_dict[a]

        return

    def load_customer_attributes(self, row, the_vertex):

        subset_dict = row[self.feature_dict['customer']].to_dict()

        for a in subset_dict:
            the_vertex[a] = subset_dict[a]

        return

    def load_support_org_attributes(self, row, the_vertex):

        subset_dict = row[self.feature_dict['support_org']].to_dict()

        for a in subset_dict:
            the_vertex[a] = subset_dict[a]

        return

    def load_vendor_attributes(self, row, the_vertex):

        subset_dict = row[self.feature_dict['vendor']].to_dict()

        for a in subset_dict:
            the_vertex[a] = subset_dict[a]

        return

    def load_num_mods(self, row, the_vertex):

        return







