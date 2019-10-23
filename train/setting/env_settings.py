import os
import sys
import yaml

def env_settings():
    config_path = os.path.join(os.path.dirname(__file__), 'environment.yaml')
    
    with open(config_path, 'rb') as f:
        conf = f.read()

    cf = yaml.load(conf)
    return cf

def get_got10k_root():
    '''
    get the path to got10k
    '''
    cf = env_settings()
   
    # print(cf.get('got10k'))
    return  cf.get('got10k')['root']

def get_got10k_train():
    '''
    get the train path to got10k
    '''
    cf = env_settings()
   
    # print(cf.get('got10k'))
    return  cf.get('got10k')['train']

def get_got10k_val():
    '''
    get the val path to got10k
    '''
    cf = env_settings()
   
    # print(cf.get('got10k'))
    return  cf.get('got10k')['val']

def get_got10k_test():
    '''
    get the test path to got10k
    '''
    cf = env_settings()
   
    # print(cf.get('got10k'))
    return  cf.get('got10k')['test']


if __name__ == "__main__":
    print(get_got10k_test())