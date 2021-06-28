import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data_FD001():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    remaining_sensors = ['s_2', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9',
                         's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def load_data_FD002():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD002.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD002.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD002.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def load_data_FD003():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD003.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD003.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD003.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def load_data_FD004():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD004.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD004.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD004.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def get_info(train_raw, test_raw):
    mm = MinMaxScaler()
    ss = StandardScaler()

    X = train_raw.iloc[:, 2:]
    idx = train_raw.iloc[:, 0:2].to_numpy()
    X_ss = ss.fit_transform(X)

    X_t = test_raw.iloc[:, 2:]
    idx_t = test_raw.iloc[:, 0:2].to_numpy()
    Xt_ss = ss.fit_transform(X_t)

    nf = X_ss.shape[1]
    ns = X_ss.shape[0]
    ns_t = Xt_ss.shape[0]

    return X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t
