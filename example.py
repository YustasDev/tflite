from datetime import datetime
from timeit import timeit

import tensorflow as tf
import tensorflow.keras.utils as tf_util



from tensorflow import compat as cmp
x = cmp.v1
from restoreTF import np as yyy
import myfile as mf
import sys
from io import TextIOWrapper, BytesIO
import os
import json

import cv2
import numpy as np
from Crypto.PublicKey import RSA
from loguru import logger





# b = yyy.arange(12)
# print(b.shape)
# d = b.reshape(3,4)
# print(d.shape)





class Sample:
    def __init__(self):
        self.a = 1
        self._b = 2
        self.__c = 3

    def __someherefunc__(self):
        self.f = 'always'
        self.__x = 'secret'
        print(self.f)
        return self.f

    def get_x(self):
        return self.__x

    def __call__(self):
        print("Instance is called via special method")


def write():
    sys.stdout.write("foobar13")

def sum_numbers(n):
    return sum(range(n+1))


if __name__ == '__main__':

    import psycopg2

    try:
        conn = psycopg2.connect(dbname='test_psycopg2', user='user1', password='password', host='localhost')
    except:
        print('Can`t establish connection to database')

    # cursor = conn.cursor()
    # cursor.execute('SELECT * FROM users')
    # all_users = cursor.fetchall()
    # print(all_users)
    # cursor.close()
    # conn.close()

    conn.autocommit = True

    with conn.cursor() as curs:
        curs.execute('SELECT * FROM users')
        all_users = curs.fetchall()
        print(all_users)

    now = datetime.now()
    name_forInsert = 'Rebecca'
    date_of_accession_forInsert = now

    value_forInsert = 51
    text_forInsert = 'zone 51'

    with conn.cursor() as curs:
        curs.execute('INSERT INTO users (name, date_of_accession) VALUES (%s, %s)', (name_forInsert, date_of_accession_forInsert))

        #curs.execute("CREATE TABLE test (id serial PRIMARY KEY, value integer, text varchar);")
        #curs.execute('INSERT INTO test (value, text) VALUES (%s, %s)',
        #             (value_forInsert, text_forInsert))
        returnsStr = curs.mogrify('INSERT INTO test (value, text) VALUES (%s, %s)',
                     (value_forInsert, text_forInsert))

        print('returnsStr: ', returnsStr)
        #curs.execute('SELECT * FROM test')

        # all_users = curs.fetchall()
        # print(all_users)

        value_forInsert = 100
        try:
            query = "SELECT * FROM test WHERE value = %s;"
            parameter = (value_forInsert,)

            with conn.cursor() as cursor:
                cursor.execute(query, parameter)
                results = cursor.fetchall()
                print(results)
        except Exception as e:
            print(f"An error occurred: {e}")



    #conn.commit()
    conn.close()








    # execution_time = timeit('sum(range(100))', number=1000)
    # print(execution_time)

    """
    key = RSA.generate(2048)

    message = b"Hello, World!"
    #message = b"Hello, World!"
    encrypted_message = key.encrypt(message, 32)
    print(encrypted_message)

    decrypted_message = key.decrypt(encrypted_message)

    # Convert the byte string to a string using the decode() method
    decoded_string = decrypted_message.decode("utf-8")

    # Print the decoded string
    print(decoded_string)

    # ===========================================>
    # new_key = RSA.generate(2048)
    # tested_message = new_key.decrypt(encrypted_message)
    # #tested_message = tested_message.decode("utf-8")
    # print('tested_message: ', tested_message)
    # ===========================================<

    logger.add("file_{time}.log", rotation="500 MB")
    logger.info(f'decoded_string: {decoded_string}')
    logger.warning("This is a warning message")
    logger.error("Error message")
    logger.opt(colors=True).warning("We got a <magenta>12345</> problem")
    logger.opt(colors=True).critical("Problem number <cyan>13</> has been identified")



    somelist = ["one", "two", "five"]

    modifyList = ([(element.count("o"), element.upper()) for element in somelist])

    print(modifyList)
    
    """




    




    # a =1
    # b = 2
    # c = 3
    #
    # lst = []
    # addlst = [a,b,c]
    # lst.extend(addlst)
    #
    # del lst[2]
    #
    # print(lst)
    #
    # obj = Sample()
    # print(dir(obj))
    #
    # model_name = '12345'
    # print('The json file for the "%s.pth" model has not been created!' % (model_name))


    # dict_of_idx_classes = {'1001': 1, '1002': 2, '1003': 3}
    # invert_dict = dict((v, k) for k, v in dict_of_idx_classes.items())
    # model_name = 'example'
    # json_file_name = model_name + '.json'
    # try:
    #     if not os.path.isfile(json_file_name):
    #         with open(json_file_name, "w") as outfile:
    #             json.dump(invert_dict, outfile)
    #     else:
    #         print('The corresponding json file already exists for the %s model', model_name)
    # except:
    #     print('Something went wrong when writing to the %s file' % (json_file_name))
    #     raise Exception('The json file was not created!')
    #
    # with open(json_file_name, "r") as clazz_file:
    #     read_json = clazz_file.read()
    #
    # clazz = json.loads(read_json)
    # values_list = list(clazz.values())
    # print(values_list)
    #




    # setup the environment
    # old_stdout = sys.stdout
    # sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)
    #
    # write()

    # get output
    # sys.stdout.seek(0)  # jump to the start
    # out = sys.stdout.read()  # read output

    # restore stdout
    # sys.stdout.close()
    # sys.stdout = old_stdout

    # do stuff with the output
    # print(out.upper())




"""
    mf.printText('the sum of the two variables will be equal to', 3, 10)
    z = 0

    # x = tf.keras.utils.load_img
    # y = tf_util.load_img
    #
    # v.app.run()
    # vtf.app.run()


    obj = Sample()
    print(dir(obj))
    print('-------------------------------------')
    print(obj.__dict__)
    print(obj.__module__)
    print(obj.__subclasshook__)
    print(obj.__someherefunc__)
    print('========================================')
    obj()


    g = obj.__someherefunc__()
    print(g)

    print(obj.get_x())
"""



