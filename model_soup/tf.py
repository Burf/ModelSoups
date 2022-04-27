import inspect
import os
import sys
import time
import numpy as np

def uniform_soup(model, path, by_name = False):
    try:
        import tensorflow as tf
    except:
        print("If you want to use 'Model Soup for Tensorflow2', please install 'tensorflow2'")
        return model
    
    if not isinstance(path, list):
        path = [path]
    soups = []
    for i, model_path in enumerate(path):
        model.load_weights(model_path, by_name = by_name)
        soup = [np.array(w) for w in model.weights]
        soups.append(soup)
    if 0 < len(soups):
        for w1, w2 in zip(model.weights, list(zip(*soups))):
            tf.keras.backend.set_value(w1, np.mean(w2, axis = 0))
    return model

def greedy_soup(model, path, data, metric, update_greedy = False, compare = np.greater_equal, by_name = False, digits = 4, verbose = True, y_true = "y_true"):
    try:
        import tensorflow as tf
    except:
        print("If you want to use 'Model Soup for Tensorflow2', please install 'tensorflow2'")
        return model
    
    if not isinstance(path, list):
        path = [path]
    score, soup = None, []
    input_key = [inp.name for inp in model.inputs]
    input_cnt = len(input_key)
    for i, model_path in enumerate(path):
        if update_greedy:
            model.load_weights(model_path, by_name = by_name)
            for w1, w2 in zip(model.weights, soup):
                tf.keras.backend.set_value(w1, np.mean([w1, w2], axis = 0))
        else:
            model = uniform_soup(model, soup + [model_path], by_name = by_name)
                
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
                    d_cnt = len(y[0])
                else:
                    x = [iter_data[k] for k in input_key if k in iter_data]
                step += 1
                #del x

                logits = model.predict(x)
                if not isinstance(logits, list):
                    logits = [logits]
                if isinstance(iter_data, dict):
                    metric_key = [key for key in inspect.getfullargspec(metric).args if key != "self"]
                    if len(metric_key) == 0:
                        metric_key = [y_true]
                    y = [iter_data[k] for k in metric_key if k in iter_data]
                    d_cnt = len(y[0])
                metric_val = np.array(metric(*(y + logits)))
                if np.ndim(metric_val) == 0:
                    metric_val = [float(metric_val)] * d_cnt
                history += list(metric_val)
                #del y, logits

                if verbose:
                    sys.stdout.write("\r[{name}] step: {step} - time: {time:.2f}s - {key}: {val:.{digits}f}".format(name = os.path.basename(model_path), step = step, time = (time.time() - start_time), key = metric.__name__ if hasattr(metric, "__name__") else str(metric), val = np.nanmean(history), digits = digits))
                    sys.stdout.flush()
            except (tf.errors.OutOfRangeError, StopIteration):
                print("")
                #gc.collect()
                break
        if 0 < len(history) and (score is None or compare(np.nanmean(history), score)):
            score = np.nanmean(history)
            if update_greedy:
                soup = [np.array(w) for w in model.weights]
            else:
                soup += [model_path]
    if len(soup) != 0:
        if update_greedy:
            for w1, w2 in zip(model.weights, soup):
                tf.keras.backend.set_value(w1, w2)
        else:
            model = uniform_soup(model, soup, by_name = by_name)
        if verbose:
            print("greedy soup best score : {val:.{digits}f}".format(val = score, digits = digits))
    return model