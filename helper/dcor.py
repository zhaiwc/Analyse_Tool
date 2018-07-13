# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:54:57 2018

@author: zhaiweichen
"""

import numpy as np
"""
Derived From:
http://jpktd.blogspot.com/2012/06/non-linear-dependence-measures-distance.html
"""
def dist(x, y):
  #1d only
  return np.abs(x[:, None] - y)

def d_n(x):
  d = dist(x, x)
  dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean() 
  return dn

def dcov_all(x, y):
  # Coerce type to numpy array if not already of that type.
  try: x.shape
  except AttributeError: x = np.array(x)
  try: y.shape
  except AttributeError: y = np.array(y)
  
  dnx = d_n(x)
  dny = d_n(y)
    
  denom = np.product(dnx.shape)
  dc = np.sqrt((dnx * dny).sum() / denom)
  dvx = np.sqrt((dnx**2).sum() / denom)
  dvy = np.sqrt((dny**2).sum() / denom)
  dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
  return dc, dr, dvx, dvy

def dcor(x,y):
  return dcov_all(x,y)[1]

if __name__ == '__main__':
    x = np.array([1,1,1,1,2])
    y = np.array([1,0,1,10,1])
    print(dcor(x,y))