import pandas as pd
from deepctr import SingleFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from model import xDeepFM_MTL
from auc_util import auroc
from keras.callbacks import Callback, EarlyStopping

ONLINE_FLAG = True
loss_weights = [0.7, 0.3, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

if __name__ == "__main__":
    data = pd.read_csv('./input/final_track1_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'], iterator=True)
    
    take=[]
    loop = True
    while loop:
        try:
            chunk=data.get_chunk(10000)
            chunk=chunk.take(list(range(100)), axis=0)
            take.append(chunk)
        except StopIteration:
            loop=False
            print('stop iteration')
    
    data = pd.concat(take, ignore_index=True)        
#     for i in range(int(data.shape[0]/10000-1)):
#         take.extend(list(range(i*100,(i+1)*100)))
#     data = data.take(take, axis=0)
    print(data.shape[0])
    
    if ONLINE_FLAG:
        test_data = pd.read_csv('./input/final_track1_test_no_anwser.txt', sep='\t', names=[
                                'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        train_size = data.shape[0]
        data = data.append(test_data)
    else:
        train_size = int(data.shape[0]*(1-VALIDATION_FRAC))

    
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
    dense_features = ['video_duration']  # 'creat_time',

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list})
    model.compile("adagrad", loss='binary_crossentropy', loss_weights=loss_weights, metrics=[auroc])
    
    my_callbacks = [EarlyStopping(monitor=auroc, min_delta=1e-6, patience=10, verbose=1, mode='max')]
    
    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=100, verbose=1,
                            callbacks=my_callbacks)
        pred_ans = model.predict(test_model_input, batch_size=2**14)

    else:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=100, verbose=1, validation_data=(test_model_input, test_labels),
                            callbacks=my_callbacks)

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
