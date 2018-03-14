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

http_verbs = [ "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH" ]
http_codes = [ i.value for i in http.HTTPStatus ]


#########
###### FIXME: path is often _missing_ -- should able able to deal with this
######

class ELBLogCreationException(Exception): pass


class DataSet:
  def __init__(self, records, fields):
    self.records = records
    self.count = len(self.records)
    self.fields = fields
    for i in self.fields:
      print('field: ', i)
      if self.fields[i] == 'string':
        setattr(self, i, one_hot([ getattr(j, i) for j in self.records]))
      else:
        setattr(self, i, np.array([[ getattr(j, i) for j in self.records]]).astype(np.float).T)

  def get_array(self):
    return np.column_stack([getattr(self, i) for i in self.fields])

  def get_probabilities(self):
    x = self.get_array()
    print("shape: ", x.shape)
    mu = x.mean(axis=0).ravel()
    # Sigma = 1/m(Sigma((x^i-mu))(x^i-mu)^T)
    sigma = np.cov(x.T)
    print("sigma: ", sigma)
    print("sigma shape: ", sigma.shape)
    #_x = x-mu.T
    #_sigma = np.dot(_x.T,_x) / (data_set.count-1)
    prob = multivariate_normal.pdf(x.T[0], mean=mu, cov=sigma)
    return prob


class ELBLog:

  """all_fields = ['httpversion', 'path', 'urihost', 'elb', 'clientport',
  'port', 'clientip', 'backend_response', 'backend_processing_time',
  'timestamp', 'message', 'received_bytes', 'geoip', 'request', 'type', 'backendport',
  'proto', '@timestamp', 'bytes', 'verb', 'backendip', 'response', '@version',
  'response_processing_time', 'request_processing_time']"""
  #excluded_items = set(all_fields) - set(included_fields)

  included_fields = {
    'received_bytes': 'int',
    # 'bytes': 'int',
    '@timestamp': 'datetime',
    'verb': 'string',
    # 'response_processing_time': 'float',
    'path': 'string',
    # 'request_processing_time': 'float',
    'response': 'string',
    'urihost': 'string',
    'elb': 'string',
    # 'backend_processing_time': 'float'
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


def one_hot(x):
  """ x is matrix of categorical values, returns
      one-hot-encoded matrix """
  enc = preprocessing.LabelEncoder()
  ohe = preprocessing.OneHotEncoder(categorical_features = [0])
  x = np.array([x]).T
  x[:,0] = enc.fit_transform(x[:,0])
  np.set_printoptions(threshold=np.nan)
  x = ohe.fit_transform(x).toarray()
  print("x: ", x[0,:])
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
  elk_gen = get_elk_records('url', 'index', 3000)
  data_set = get_data_set(elk_gen)
  print("count: ", data_set.count)
  print("prob: ", data_set.get_probabilities())


if __name__ == '__main__':
  main()
