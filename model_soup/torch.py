import inspect
import os
import sys
import time
import numpy as np

def uniform_soup(model, path, device = "cpu", by_name = False):
    try:
        import torch
    except:
        print("If you want to use 'Model Soup for Torch', please install 'torch'")
        return model
        
    if not isinstance(path, list):
        path = [path]
    model = model.to(device)
    model_dict = model.state_dict()
    soups = {key:[] for key in model_dict}
    for i, model_path in enumerate(path):
        weight = torch.load(model_path, map_location = device)
        weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight
        if by_name:
            weight_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
        for k, v in weight_dict.items():
            soups[k].append(v)
    if 0 < len(soups):
        soups = {k:(torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype) for k, v in soups.items() if len(v) != 0}
        model_dict.update(soups)
        model.load_state_dict(model_dict)
    return model

def greedy_soup(model, path, data, metric, device = "cpu", update_greedy = False, compare = np.greater_equal, by_name = False, digits = 4, verbose = True, y_true = "y_true"):
    try:
        import torch
    except:
        print("If you want to use 'Model Soup for Torch', please install 'torch'")
        return model

    if not isinstance(path, list):
        path = [path]
    score, soup = None, []
    model = model.to(device)
    model.eval()
    model_dict = model.state_dict()
    input_key = [key for key in inspect.getfullargspec(model.forward).args if key !=  "self"]
    input_cnt = len(input_key)
    for i, model_path in enumerate(path):
        if update_greedy:
            weight = torch.load(model_path, map_location = device)
            weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight
            if by_name:
                weight_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
            if len(soup) != 0:
                weight_dict = {key:(torch.sum(torch.stack([weight_dict[key], soup[key]]), axis = 0) / 2).type(weight_dict[key].dtype)  for key in model_dict.keys()}
            model_dict.update(weight_dict)
            model.load_state_dict(model_dict)
        else:
            model = uniform_soup(model, soup + [model_path], device = device, by_name = by_name)
                
        iterator = iter(data)
        history = []
        step = 0
        start_time = time.time()
        while True:
            try:
                text = ""
                iter_data = next(iterator)
                if not isinstance(iter_data, dict):
                    x = iter_data[:input_cnt]
                    y = list(iter_data[input_cnt:])
                    y = [d.to(device) if isinstance(d, torch.Tensor) else d for d in y]
                    d_cnt = len(y[0])
                else:
                    x = [iter_data[k] for k in input_key if k in iter_data]
                x = [d.to(device) if isinstance(d, torch.Tensor) else d for d in x]
                step += 1
                #del x

                with torch.no_grad():
                    logits = model(*x)
                    if isinstance(logits, torch.Tensor):
                        logits = [logits]
                        
                    if isinstance(iter_data, dict):
                        metric_key = [key for key in inspect.getfullargspec(func).args if key !=  "self"]
                        if len(metric_key) == 0:
                            metric_key = [y_true]
                        y = [iter_data[k] for k in metric_key if k in iter_data]
                        y = [d.to(device) if isinstance(d, torch.Tensor) else d for d in y]
                        d_cnt = len(y[0])
                    metric_val = np.array(metric(*(y + logits)))
                    if np.ndim(metric_val) == 0:
                        metric_val = [float(metric_val)] * d_cnt
                    history += list(metric_val)
                    #del y, logits

                if verbose:
                    sys.stdout.write("\r[{name}] step: {step} - time: {time:.2f}s - {key}: {val:.{digits}f}".format(name = os.path.basename(model_path), step = step, time = (time.time() - start_time), key = metric.__name__ if hasattr(metric, "__name__") else str(metric), val = np.nanmean(history), digits = digits))
                    sys.stdout.flush()
            except StopIteration:
                print("")
                #gc.collect()
                break
        if 0 < len(history) and (score is None or compare(np.nanmean(history), score)):
            score = np.nanmean(history)
            if update_greedy:
                soup = weight_dict
            else:
                soup += [model_path]
    if len(soup) != 0:
        if update_greedy:
            model_dict.update(soup)
            model.load_state_dict(model_dict)
        else:
            model = uniform_soup(model, soup, device = device, by_name = by_name)
        if verbose:
            print("greedy soup best score : {val:.{digits}f}".format(val = score, digits = digits))
    return model