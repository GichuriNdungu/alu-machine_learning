#!/usr/bin/env python3
def summation_i_squared(n):
     if type(n) != int:
          return None
     else:
        squared_num = n*((n+1)*(2*n+1))//6
        return squared_num
