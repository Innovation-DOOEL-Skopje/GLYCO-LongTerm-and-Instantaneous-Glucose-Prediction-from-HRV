import keras.backend as K

# ____________________________________________________________________________________________________________________________________
def true_positives(y_test, y_pred):
    return K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))

def false_positives(y_test, y_pred):
    return K.sum(K.abs(y_test - 1) * y_pred)

def false_negatives(y_test, y_pred):
    return K.sum(y_test * K.abs(y_pred - 1))

def true_negatives(y_test, y_pred):
    return K.sum((y_test - 1) * (y_pred - 1))

# ____________________________________________________________________________________________________________________________________

def precision_func(y_test, y_pred):
    true_positives = K.sum(y_test * y_pred)
    false_positives = K.sum(K.abs(y_test - 1) * y_pred)
    return true_positives / (true_positives + false_positives + K.epsilon())

def recall_func(y_test, y_pred):
    true_positives = K.sum(y_test * y_pred)
    false_negatives = K.sum(y_test * K.abs(y_pred - 1))
    return true_positives / (true_positives + false_negatives + K.epsilon())

def f1_1_func(y_test, y_pred):
    precision = precision_func(y_test, y_pred)
    recall = recall_func(y_test, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# ____________________________________________________________________________________________________________________________________

def specificity_func(y_test, y_pred):
    true_negatives = K.sum((y_test - 1) * (y_pred - 1))
    false_positives = K.sum(K.abs(y_test - 1) * y_pred)
    return true_negatives / (true_negatives + false_positives + K.epsilon())

def npv_func(y_test, y_pred):
    true_negatives = K.sum((y_test - 1) * (y_pred - 1))
    false_negatives = K.sum(y_test * K.abs(y_pred - 1))
    return true_negatives / (true_negatives + false_negatives + K.epsilon())

def f1_0_func(y_test, y_pred):
    specificity = specificity_func(y_test, y_pred)
    npv = npv_func(y_test, y_pred)
    return 2 * (specificity * npv) / (specificity + npv + K.epsilon())

# ____________________________________________________________________________________________________________________________________

def f1_macro_func(y_test, y_pred):
    f1_1 = f1_1_func(y_test, y_pred)
    f1_0 = f1_0_func(y_test, y_pred)
    return (f1_1 + f1_0) / 2
