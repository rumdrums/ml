from __future__ import print_function
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import pprint
import datetime
import dateutil.parser
import http
from scipy.stats import multivariate_normal
import numpy as np
from sklearn import preprocessing
import pandas as pd

http_verbs = [ "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH" ]
http_codes = [ i.value for i in http.HTTPStatus ]


#########
###### FIXME: path is often _missing_ -- should able able to deal with this
######        datetime needs to be fixed
#########

class ELBLogCreationException(Exception): pass

class DataSet:
  def __init__(self, records, fields):
    self.records = records
    self.count = len(self.records)
    self.fields = fields
    for i in self.fields:
      print('field: ', i)
      if self.fields[i] == 'string':
        setattr(self, i, pd.get_dummies([ getattr(j, i) for j in self.records]))
      else:
        setattr(self, i, pd.DataFrame({ i: [ getattr(j, i) for j in self.records]}))

  def get_array(self):
    return pd.concat( [ getattr(self, i) for i in self.fields ], axis=1)

  def get_probabilities(self):
    x = self.get_array()
    print("shape: ", x.shape)
    mu = x.mean(axis=0)
    # Sigma = 1/m(Sigma((x^i-mu))(x^i-mu)^T)
    sigma = np.cov(x, rowvar=False)
    # print("sigma: ", sigma)
    print("sigma shape: ", sigma.shape)
    #_x = x-mu.T
    #_sigma = np.dot(_x.T,_x) / (data_set.count-1)
    return multivariate_normal.pdf(x, mean=mu, cov=sigma)


class ELBLog:

  """all_fields = ['httpversion', 'path', 'urihost', 'elb', 'clientport',
  'port', 'clientip', 'backend_response', 'backend_processing_time',
  'timestamp', 'message', 'received_bytes', 'geoip', 'request', 'type', 'backendport',
  'proto', '@timestamp', 'bytes', 'verb', 'backendip', 'response', '@version',
  'response_processing_time', 'request_processing_time']"""
  #excluded_items = set(all_fields) - set(included_fields)

  included_fields = {
    'received_bytes': 'float',
    'bytes': 'float',
    '@timestamp': 'float',
    'verb': 'string',
    'response_processing_time': 'float',
    'path': 'string',
    'request_processing_time': 'float',
    'response': 'string',
    'urihost': 'string',
    'elb': 'string',
    'backend_processing_time': 'float'
  }

  @property
  def timestamp(self):
    return self._timestamp

  @timestamp.setter
  def timestamp(self, t):
    self._timestamp = t

  @property
  def path(self):
    return self._path

  @path.setter
  def path(self, p):
    # simplify it down to first element of path
    self._path = p.split('/')[1]

  def __init__(self, src):
    for i in ELBLog.included_fields:
      if i not in src:
        print("%s not in record!!!" % i)
        raise ELBLogCreationException
      if i == '@timestamp':
        setattr(self, i, dateutil.parser.parse(src[i]).timestamp())
      else:
        setattr(self, i, src[i])
    self.timestamp = getattr(self, '@timestamp')


def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


def one_hot(x):
  """ broken """
  enc = preprocessing.LabelEncoder()
  ohe = preprocessing.OneHotEncoder()
  x = x.flatten()
  x = enc.fit_transform(x)
  np.set_printoptions(threshold=np.nan)
  x = ohe.fit_transform(x.reshape(-1, 1)).toarray()
  return x


def unix_milli(t):
  epoch = datetime.datetime.utcfromtimestamp(0)
  return (t - epoch).total_seconds() * 1000.0


def get_elk_records(url, index_pattern, n_mins_ago=None):
  """ return generator """

  es = Elasticsearch(url, verify_certs=False)


  if n_mins_ago is None:
    from_time = 0
  else:
    from_time = unix_milli(datetime.datetime.now() - datetime.timedelta(minutes=n_mins_ago))

  to_time = unix_milli(datetime.datetime.now())

  request = """{
    "version": true,
    "sort": [
      {
        "@timestamp": {
          "order": "desc",
          "unmapped_type": "boolean"
        }
      }
    ],
    "stored_fields": [
      "_source"
    ],
    "script_fields": {},
    "docvalue_fields": [
      "@timestamp"
    ],
    "query": {
      "bool": {
        "must": [
          {
            "query_string": {
              "query": "NOT tags: _grokparsefailure",
              "analyze_wildcard": true,
              "default_field": "*"
            }
          },
          {
            "range": {
              "@timestamp": {
                "gte": %s,
                "lte": %s,
                "format": "epoch_millis"
              }
            }
          }
        ]
      }
    }
  }""" % (from_time, to_time)

  gen = helpers.scan(es,
    index=index_pattern,
    query=json.loads(request),
    _source_include=[ i for i in ELBLog.included_fields],
    scroll='2m',
    size=10000)

  return gen


def parse_http_code(_code):
  code = int(_code)
  if code in http_codes:
    return code
  # if not standard, round down to nearest 100
  return math.floor(code/100) * 100


def get_data_set(gen):
  data_set = []
  for i in gen:
    try:
      data_set.append(ELBLog(i['_source']))
    except ELBLogCreationException:
      print("Failed to create record: ", i)
  return DataSet(data_set, ELBLog.included_fields)


def main():
  elk_gen = get_elk_records("http://localhost:9004", "hm-sys-lb-*", 30)
  data_set = get_data_set(elk_gen)
  print("count: ", data_set.count)
  print("prob: ", [ i for i in data_set.get_probabilities() ])


if __name__ == '__main__':
  main()
