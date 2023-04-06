from yaml import load
from yaml import FullLoader
import tensorflow as tf

def set_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)
    
def get_len_size(LAG,x_size):
  return int(x_size/LAG) * LAG