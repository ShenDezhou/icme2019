import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr import SingleFeat
import tensorflow as tf
from keras.callbacks import EarlyStopping
import gc

loss_weights = [0.81, 1.01, ] 

def model_pool(defaultfilename='./input/final_track1_train.txt', defaulttestfile='./input/final_track1_test_no_anwser.txt',
                defaultcolumnname=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'],
                defaulttarget=['finish', 'like'], defaultmodel="AFM", PERCENT=100):
        
    sparse_features=[]
    dense_features=[]
    target=defaulttarget    
     
    #1 train file
    data = pd.read_csv(defaultfilename, sep='\t', names=defaultcolumnname, iterator=True)
    #1 train file concats
    take=[]
    loop = True
    while loop:
        try:
            chunk=data.get_chunk(10000000)
            chunk=chunk.sample(frac=PERCENT/100., replace=True, random_state=1)
            take.append(chunk)
            gc.collect()
        except StopIteration:
            loop=False
            print('stop iteration')
            
    data = pd.concat(take, ignore_index=True, copy=False) 
    train_size = data.shape[0]
    print(train_size)
    take.clear()
    del [chunk,take]
        
    for column in data.columns:
        if column in defaulttarget:
            continue
        if data[column].dtype in  [numpy.float_ , numpy.float64]:
            dense_features.append(column)
        if data[column].dtype in [numpy.int_, numpy.int64]:
            sparse_features.append(column)
            
#     sparse_features=list(set(sparse_features))
#     dense_features=list(set(dense_features))
    #***************normal
    #3. Remove na values
    data[sparse_features].fillna('-1', inplace=True)
    data[dense_features].fillna(0, inplace=True)
    
    #4. Label Encoding for sparse features, and do simple Transformation for dense features
    labelencoder={}
    for feat in sparse_features:
        lbe = LabelEncoder()
        labelencoder[feat]=lbe
        data[feat] = lbe.fit_transform(data[feat])
    #5. Dense normalize
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
    #*****************normal
        #6. generate input data for model
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    
    #****************model
    # 6.choose a model
    import pkgutil
    import mdeepctr.models
#     modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.__path__)]
#     modelname = input("choose a model: "+",".join(modelnames)+"\n")
#     if not modelname:
    modelname=defaultmodel
    # 7.build a model
    model = getattr(mdeepctr.models, modelname)({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, final_activation='sigmoid', output_dim=len(defaulttarget))
    # 8. eval predict
    def auc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    
    model.compile("adam", loss="binary_crossentropy", loss_weights=loss_weights, metrics=[auc])


    train_model_input = [data[feat.name].values for feat in sparse_feature_list] + \
                        [data[feat.name].values for feat in dense_feature_list]
    train_labels = [data[target].values for target in defaulttarget]

    my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-2, patience=1, verbose=1, mode='min')]

    history = model.fit(train_model_input, train_labels,
                batch_size=2**14, epochs=3, verbose=1, callbacks=my_callbacks)

    del [train_model_input, train_labels, data]
#     import objgraph
#     objgraph.show_refs([data], filename='data-graph.png')
    
    #2 extract file       
    test_data = pd.read_csv(defaulttestfile, sep='\t', names=defaultcolumnname, )
    raw_test_data=test_data.copy()
    #data = data.append(test_data)
    test_size=test_data.shape[0]
    print(test_size)
    #***************normal
    #3. Remove na values
    test_data[sparse_features].fillna('-1', inplace=True)
    test_data[dense_features].fillna(0, inplace=True)
    #4. Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
#         lbe = LabelEncoder()
        lbe = labelencoder[feat]
        test_data[feat] = lbe.fit_transform(test_data[feat])
    #5. Dense normalize
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        test_data[dense_features] = mms.fit_transform(test_data[dense_features])
    #*****************normal
    #extract = test_data
    test_model_input = [test_data[feat.name].values for feat in sparse_feature_list] + \
        [test_data[feat.name].values for feat in dense_feature_list]
       
    pred_ans = model.predict(test_model_input, batch_size=2**14)
        
    result = raw_test_data[['uid', 'item_id', 'finish', 'like']].copy()
    result.rename(columns={'finish': 'finish_probability',
                           'like': 'like_probability'}, inplace=True)
    result['finish_probability'] = pred_ans[0]
    result['like_probability'] = pred_ans[1]
    output = "%s-result.csv" % (modelname)
    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
        output, index=None, float_format='%.6f')
    
    return history

if __name__ == "__main__":
    import pkgutil
    import mdeepctr.models
    modelnames = [name for _, name, _ in pkgutil.iter_modules(mdeepctr.models.__path__)]
    functions = ["AFM", "DCN", "MLR",  "DeepFM",
           "MLR", "NFM", "DIN", "FNN", "PNN", "WDL", "xDeepFM", "AutoInt", ]
    models_dic = dict((function.lower(),function) for function in functions)
    for modelname in modelnames:
        print(modelname)
        if models_dic[modelname] not in ["AutoInt"]:
            continue
        history = model_pool(defaultmodel=models_dic[modelname], PERCENT=10)
        print(history.history)