#%%
import inspect
import json
import argparse
import sys
import pdb
import torch
import monai
import os
import mlflow
from monai.transforms import *
# from monai.data import *
from pydoc import locate
from ignite.metrics import Accuracy, Loss
from ignite.engine import _prepare_batch,create_supervised_evaluator,Events
#%%
# pass args named with a.b, get a nested dict
def unflatten(dictobj):
    res = {}
    for k in dictobj:
        lev = res
        parts = k.split('.')
        for part in parts[:-1]:            
            lev[part]={}
            lev = lev[part]
        lev[parts[-1]]=dictobj[k]
    return res
#%%
# vars(args) will mostly do
# def props(obj):
#     # Argparse to json
#     pr = {}
#     for name in dir(obj):
#         value = getattr(obj, name)
#         if not name.startswith('__') and not inspect.ismethod(value):
#             pr[name] = value
#     return pr
#%%
class ArgParser:
    """Takes a flat json file specifying argument names - eg argnames.json, 
        and creates argparse object. Provides parse_args, returning command line args as dict
    """
    def __init__(self,jsonfile,title=None):
        self.parser = argparse.ArgumentParser(description=title,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('cfgname')
        for key, val in json.load(open(jsonfile)).items():
            ar = ['--%s' % key]
            kwar = val
            if 'default' in val:
                kwar['type'] = type(val['default'])
            self.parser.add_argument(*ar,**kwar)
    def parse_args(self,argsin):
        # returns as flat dict
        return vars(self.parser.parse_args(argsin))
#%% Training dataset (data_transform are hard-coded, not taking from json file. Need to fix)
def full_dataset(path,dtype,vol_size,keys,transforms):
    obj = locate(dtype)
    if obj is None:
        raise Exception("instantiate: typename not known: %s" % dtype)
    lgg_label = 0
    hgg_label = 1
    data_path = path+"HGG"
    folder_names = os.listdir(data_path)
    file_names = []
    for j in range(75):
        file_names.append({'img':data_path+"/"+folder_names[j]+"/"+folder_names[j]+"_t1.nii.gz",'label':hgg_label})
    data_path = path+"LGG"
    folder_names = os.listdir(data_path)
    for j in range(75):
        file_names.append({'img':data_path+"/"+folder_names[j]+"/"+folder_names[j]+"_t1.nii.gz",'label':lgg_label})
    train_file_names = file_names[:65]+file_names[75:-10]
    valid_file_names = file_names[65:75]+file_names[-10:]
    net_orientation = "RAS"
    data_transform = Compose([
                            LoadNiftiD(keys = keys[0]),
                            AddChannelD(keys = keys[1]),
                            # SpacingD(KEYS, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
                            OrientationD(keys = keys[2], axcodes = net_orientation),
                            ScaleIntensityD(keys = keys[3]),
                            ResizeD(keys = keys[4], spatial_size = vol_size),
                            # RandAffineD(KEYS, spatial_size=(-1, -1, -1),rotate_range=(0, 0, np.pi/2),scale_range=(0.1, 0.1),mode=('bilinear', 'nearest'),prob=1.0),
                            ToTensorD(keys = keys[5]),
                            ])
    print("Reading the training data:")
    dict_train_dataset = obj(train_file_names,data_transform)
    print("Reading the validation data:")
    dict_valid_dataset = obj(valid_file_names,data_transform)
    return dict_train_dataset,dict_valid_dataset
#%% Dataloader for training and validation
def loader(train_set,valid_set,typename,train_kwargs = None,valid_kwargs = None):
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: typename not known: %s" % typename)
    if train_kwargs is None:
        train_loader = obj(train_set,batch_size = 3, shuffle = True, num_workers = 1)
    else:
        train_loader = obj(train_set,**train_kwargs)
    if valid_kwargs is None:
        valid_loader = obj(valid_set,batch_size = 2, shuffle = True, num_workers = 1)
    else:
        valid_loader = obj(valid_set,**valid_kwargs)
    return train_loader,valid_loader
#%% Model instance
def instantiate(typename, args=None, kwargs=None):
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: Modelname not known: %s" % typename)
    if args is None and kwargs is None:
        return obj()
    if args is not None:
        if kwargs is None:
            inst = obj(*args)
        else:
            inst = obj(*args,**kwargs)        
    else:
        inst = obj(**kwargs)
    return inst
#%% Optimizer instance
def weight_update_inst(typename, model, args=None, kwargs=None):
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: Optimizer not known: %s" % typename)
    if args is None and kwargs is None:
        return obj(model.parameters())
    if args is not None:
        if kwargs is None:
            inst = obj(model.parameters(),*args)
        else:
            inst = obj(model.parameters(),*args,**kwargs)        
    else:
        inst = obj(model.parameters(),**kwargs)
    return inst
#%% Criterion instance
def loss_func_inst(typename, args=None, kwargs=None):
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: Criterion not known: %s" % typename)
    if args is None and kwargs is None:
        return obj()
    if args is not None:
        if kwargs is None:
            inst = obj(*args)
        else:
            inst = obj(*args,**kwargs)        
    else:
        inst = obj(**kwargs)
    return inst
def engine_inst(typename,device,train_loader,valid_loader,my_model,inferer,opt,loss_func,train_handlers,args = None,kwargs = None):
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: Engine not known: %s" % typename)
    def prep_batch(batch, device = device, non_blocking = False):
        return _prepare_batch((batch["img"],torch.squeeze(batch["label"],1).type(torch.LongTensor)), device, non_blocking)
    if args is None and kwargs is None:
        return obj(device = device,train_data_loader = train_loader,network = my_model,inferer = inferer,optimizer = opt,loss_function = loss_func,train_handlers = train_handlers,prepare_batch = prep_batch)
    if args is not None:
        if kwargs is None:
            inst = obj(device = device,prepare_batch = prep_batch,train_data_loader = train_loader,network = my_model,inferer = inferer,optimizer = opt,loss_function = loss_func,train_handlers = train_handlers,*args)
        else:
            inst = obj(device = device,prepare_batch = prep_batch,train_data_loader = train_loader,network = my_model,inferer = inferer,optimizer = opt,loss_function = loss_func,train_handlers = train_handlers,*args,**kwargs)        
    else:
        inst = obj(device = device,prepare_batch = prep_batch,train_data_loader = train_loader,network = my_model,inferer = inferer,optimizer = opt,loss_function = loss_func,train_handlers = train_handlers,**kwargs)
    evaluator = create_supervised_evaluator(my_model,metrics={'accuracy': Accuracy(),'nll': Loss(F.cross_entropy)},prepare_batch = prep_batch,device=device)
    @inst.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy, avg_nll))
        mlflow.log_metric("train_accuracy",avg_accuracy)        
        mlflow.log_metric("train_loss",avg_nll)
    @inst.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy, avg_nll))
        mlflow.log_metric("valid_accuracy",avg_accuracy)
        mlflow.log_metric("valid_loss",avg_nll)
    return inst
#%%
class JsonConfig:
    def __init__(self,filename):
        self.config = json.load(open(filename))
    def __repr__(self):
        return str(self.config)
    def update_from_args(self,args):
        self.config.update(unflatten(args))
    def get_dataset(self):
        data_type = self.config["DATASET"]["DATA_TYPE"]
        data_path = self.config["DATASET"]["DATASET_PATH"]
        vol_shape = self.config["DATASET"]["VOLUME_SHAPE"]
        keys = self.config['DATASET']['TRANSFORMS_KEYS']
        transforms = self.config['DATASET']['TRANSFORMS_DICT']
        vol_shape = tuple(vol_shape)
        return full_dataset(data_path,data_type,vol_shape,keys,transforms)
    def get_loaders(self,train_set,valid_set):
        typename = self.config['LOADER']['LOADER_TYPE']
        train_kwargs = None
        valid_kwargs = None
        if 'LOADER_ARGS' in self.config['LOADER']:
            train_kwargs = self.config['LOADER']["LOADER_ARGS"]["Train"]
            valid_kwargs = self.config['LOADER']["LOADER_ARGS"]["Valid"]
        return loader(train_set,valid_set,typename,train_kwargs,valid_kwargs)
    def get_model_instance(self):
        typename = self.config['MODEL']['MODEL_TYPE']
        kwargs = None
        if 'MODEL_ARGS' in self.config['MODEL']:
            kwargs = self.config['MODEL']['MODEL_ARGS']
        return instantiate(typename, None, kwargs)
    def get_optimizer_instance(self,model):
        typename = self.config['OPTIMIZER']['OPTIMIZER_TYPE']
        kwargs = None
        if "OPTIMIZER_ARGS" in self.config['OPTIMIZER']:
            kwargs = self.config["OPTIMIZER"]["OPTIMIZER_ARGS"]
        return weight_update_inst(typename, model, None, kwargs)
    def get_criterion_instance(self):
        typename = self.config['CRITERION']['CRITERION_TYPE']
        kwargs = None
        if "CRITERION_ARGS" in self.config['CRITERION']:
            kwargs = self.config["CRITERION"]["CRITERION_ARGS"]
        return loss_func_inst(typename, None, kwargs)
    def get_train_engine(self,device,train_loader,valid_loader,my_model,inferer,opt,loss_func,train_handlers):
        typename = self.config['TRAIN_ENGINE']['ENGINE_TYPE']
        kwargs = None
        if "ENGINE_ARGS" in self.config['TRAIN_ENGINE']:
            kwargs = self.config['TRAIN_ENGINE']['ENGINE_ARGS']
        return engine_inst(typename,device,train_loader,valid_loader,my_model,inferer,opt,loss_func,train_handlers,None,kwargs)
#%%
# def parseInputs(argsin=sys.argv,title=None):
def parseInputs(argsin = ["monai_train.py","argnames.json"],title=None):
    """for command line parsing first argument is always cfg json filename. overrides follow with --<group>.<param>"""
    my_args_instance = ArgParser(argsin[1],title)
    my_args = my_args_instance.parse_args(["exploration/monai_abc.json"])
    cfg = None
    if 'cfgname' in my_args:
        cfg = JsonConfig(my_args['cfgname']) 
    return cfg, my_args