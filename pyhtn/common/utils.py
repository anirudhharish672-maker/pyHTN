from random import choices as sample_w_replacement
import string
import hashlib
from base64 import b64encode
import numpy as np

# ------------------------------------------------------
# : rand_uid()

alpha_num_chars = string.ascii_letters + string.digits
def rand_uid(k=30):
    '''A more compact random string than uuid(). Includes capital letters, 
       With k>=22 has less chance of hash conflicts than uuid(). 
       With k=22 space of 62^22 = 2.7e+39 > 2^128 = 3.4e38 for uuid.
       With k=30 space of 62^30 = 5.9e+53 > 2^128 = 3.4e38 for uuid.
    '''
    return ''.join(sample_w_replacement(alpha_num_chars, k=k))


# ------------------------------------------------------
# : unique_hash()

def update_unique_hash(m,obj):
    ''' Recrusive depth-first-traversal to buildup hash '''

    if(isinstance(obj,str)):
        m.update(obj.encode('utf-8'))
    elif(isinstance(obj,(tuple,list, np.ndarray))):
        for i,x in enumerate(obj):
            update_unique_hash(m,i)
            update_unique_hash(m,x)
    elif(isinstance(obj,dict)):
        for k,v in obj.items():
            update_unique_hash(m,k)
            update_unique_hash(m,v)
    elif(isinstance(obj,bytes)):
        m.update(obj)
    else:
        m.update(str(obj).encode('utf-8'))


def unique_hash(stuff, hash_func='sha1'):
    '''Returns a 64-bit encoded hashstring of heirachies of basic python data'''
    m = hashlib.new(hash_func)
    update_unique_hash(m,stuff) 

    # Encode in base64 map the usual altchars '/' and "+' to 'A' and 'B".
    s = b64encode(m.digest(),altchars=b'AB').decode('utf-8')
    # Strip the trailing '='.
    s = s[:-1]
    return s


