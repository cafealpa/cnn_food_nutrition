import time

def connection_api(food_name, food_db):
    time.sleep(5)
    return_dict = {
        'name': food_name,
        'db': food_db
    }

    return return_dict