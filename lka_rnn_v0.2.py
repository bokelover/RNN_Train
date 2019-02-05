import tensorflow as tf
import numpy as np
from scipy.stats import norm
from scipy import signal
import random
import os
from tensorflow.contrib.tensorboard.plugins import projector

"""
単層RNN(LSTM)を用いてレーンキープ制御を解かせる
教師データ：CSVデータ

18.11.23 ~ 30 Bokelover

# データ列入力順 [単位]
０　タイムスタンプ
１　左白線位置[m]
２　右白線位置[m]
３　カメラ検出ヨー角[rad]
４　カーブ曲率[1/m]
５　車速[km/h]
６　斜めx加速度[m/s/s]
７　斜めy加速度[m/s/s]
8  ヨーレート[deg/sec]
9　舵角[rad]
10 500ms後の舵角[rad]
11 10の[deg]
"""


# >================== 関数 ================== <
def create_data(num_of_samples, input_sensors):

    # 出力の用意
    D_X = np.zeros((num_of_samples, input_sensors))
    r_t = np.zeros(num_of_samples)
    # カウンター
    ct = 1
    lt = 0
    tmp_lp = 0

    for row_idx in range(1, num_of_samples):
        yoko_P = L_pos[row_idx] + R_pos[row_idx]
        D_X[row_idx, 0] = L_pos[row_idx] / 2.5
        D_X[row_idx, 1] = R_pos[row_idx] / 2.5
        D_X[row_idx, 2] = yaw_ang[row_idx] / 0.05
        D_X[row_idx, 3] = curv_R[row_idx] / 0.05
        D_X[row_idx, 4] = velocity[row_idx] / 100  # [km/h]
        D_X[row_idx, 5] = tate_G[row_idx] / 5  # [m/s]
        D_X[row_idx, 6] = yoko_G[row_idx] / 5  # [m/s]
        D_X[row_idx, 7] = yaw[row_idx] / 5
        D_X[row_idx, 8] = pow(yoko_P, 3) * 100
        D_X[row_idx, 9] = pow(yoko_P, 5) * 10000
        if tmp_lp - yoko_P != 0:
            lat_v = (tmp_lp - yoko_P) / (ct * 0.015)
            lt = ct
            ct = 1
        elif tmp_lp == 0:
            pass
        else:
            ct += 1
        if lt > 0:
            D_X[row_idx, 10] = lat_v
            lt - 1
        else:
            D_X[row_idx, 10] = 0
        D_X[row_idx, 11] = yoko_v[row_idx] / 0.4  # [km/h]
        D_X[row_idx, 12] = suberi[row_idx] / 5.6  # [m/s]
        D_X[row_idx, 13] = ttf05[row_idx] / 0.4  # [m/s]
        D_X[row_idx, 14] = ttf08[row_idx] / 1.3

        tmp_lp = yoko_P
        r_t[row_idx] = (str_a[row_idx]) / 1.507
    print('training_data_regularize... OK!')
    # np.save('./training_X_2.npy', D_X)
    # np.save('./training_t_2.npy', r_t)
    return D_X, r_t


# テストデータ作成
def create_data_a(num_of_samples, input_sensors):
    # 出力の用意
    D_X = np.zeros((num_of_samples, input_sensors))
    r_t = np.zeros(num_of_samples)
    ct = 1
    lt = 0
    tmp_lp = 0

    for row_idx in range(num_of_samples):
        yoko_P_a = L_pos_a[row_idx] + R_pos_a[row_idx]
        D_X[row_idx, 0] = L_pos_a[row_idx] / 2.5
        D_X[row_idx, 1] = R_pos_a[row_idx] / 2.5
        D_X[row_idx, 2] = yaw_ang_a[row_idx] / 0.05
        D_X[row_idx, 3] = curv_R_a[row_idx] / 0.05
        D_X[row_idx, 4] = velocity_a[row_idx] / 100  # [km/h]
        D_X[row_idx, 5] = tate_G_a[row_idx] / 5  # [m/s]
        D_X[row_idx, 6] = yoko_G_a[row_idx] / 5  # [m/s]
        D_X[row_idx, 7] = yaw_a[row_idx] / 5
        D_X[row_idx, 8] = pow(yoko_P_a, 3) * 100
        D_X[row_idx, 9] = pow(yoko_P_a, 5) * 10000

        if tmp_lp - yoko_P_a != 0:
            lat_v = (tmp_lp - yoko_P_a) / (ct * 0.015)
            print(lat_v)
            lt = ct
            ct = 1

        elif tmp_lp == 0:
            pass

        else:
            ct += 1

        if lt > 0:
            D_X[row_idx, 10] = lat_v
            lt - 1
            print(lat_v)
        else:
            D_X[row_idx, 10] = 0

        D_X[row_idx, 11] = yoko_v_a[row_idx] / 0.4  # [km/h]
        D_X[row_idx, 12] = suberi_a[row_idx] / 5.6  # [m/s]
        D_X[row_idx, 13] = ttf05_a[row_idx] / 0.4  # [m/s]
        D_X[row_idx, 14] = ttf08_a[row_idx] / 1.3

        tmp_lp = yoko_P_a
        r_t[row_idx] = (str_a_a[row_idx]) / 1.507
    # np.save('./testing_X_3".npy', D_X)
    # np.save('./testing_t_3.npy', r_t)
    print('testing_data_regularize... OK!')
    return D_X, r_t


# データシャッフル関数
def shuffle(X, ts, num_data):
    data_range = len(X) - num_data - 1
    random_range = np.random.randint(0, data_range)
    x_s = np.zeros((num_data, max_length_of_seaquence))
    t_s = np.zeros((num_data))
    j = 0
    for i in range(random_range, random_range + num_data):
        x_s[j] = X[i]
        t_s[j] = ts[i]
        j += 1
    return x_s, t_s


# バッチの生成
def get_batch(batch_size, X, t, count):
    data_range = len(X) - batch_size - 1
    random_number = np.random.randint(0, int(data_range / batch_size) - 1)
    random_range = np.random.randint(0, 10)
    if (random_number + 1) * 10 > data_range:

        print('ERROR!')
    else:
        elect_num = random_number * 10 + random_range

    x_s = np.zeros((batch_size, max_length_of_seaquence))
    t_s = np.zeros((batch_size))
    j = 0
    for i in range(elect_num, elect_num + batch_size):
        x_s[j] = X[i]
        t_s[j] = t[i]
        j += 1

    xs = np.array([[[y] for y in list(x_s[r])] for r in range(0, batch_size)])
    ts = np.array([[t_s[r]] for r in range(0, batch_size)])
    return xs, ts, count


# 未知のデータセット作成
def make_prediction(X, Y, num_of_prediction_data):
    xs, ts = shuffle(X, Y, num_of_prediction_data)
    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])


# 推論・学習するモデルの定義
def inference(x, istate_ph):
    with tf.name_scope("RNN_inference") as scope:

        weight1_var = tf.Variable(tf.truncated_normal(
            [Num_input_node, Num_hidden_node], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal(
            [Num_hidden_node, Num_output_str], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([Num_hidden_node], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([Num_output_str], stddev=0.1), name="bias2")
        in1 = tf.transpose(x, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, Num_input_node])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(in3, max_length_of_seaquence, 0)

        cell = tf.nn.rnn_cell.BasicLSTMCell(Num_hidden_node, forget_bias=forget_rate, state_is_tuple=False)
        # Dropout
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=1.0, state_keep_prob=1.0)
        rnn_output, states_op = tf.nn.static_rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        results = [weight1_var, weight2_var, bias1_var, bias2_var]
        return output_op, states_op, results


# 結果出力
def calc_accuracy(output_op, X_a, t_a, prints=False):
    inputs, ts = make_prediction(X_a, t_a, num_of_prediction_epochs)
    # 推論実行
    pred_dict = {
        preds: inputs,
        Y: ts,
        istate_ph: np.zeros((num_of_prediction_epochs, Num_hidden_node * 2)),
    }
    output = sess.run([output_op], feed_dict=pred_dict)

    def print_result(i, p, q):
        print("output: %f, correct: %f, error: %f" % (
            p * 1.507 * 180 / np.pi, q * 1.507 * 180 / np.pi, abs(q - p) * 1.507 * 180 / np.pi))

    if prints:
        [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ts)]

    opt = abs(output - ts)[0]
    total = abs(sum((ts - opt)))
    print(total / num_of_prediction_epochs * 1.507 * 180 / np.pi)
    print(sum(abs(ts)) / num_of_prediction_epochs * 1.507 * 180 / np.pi)
    print("error mean: %f, Max error: %f" % (
        np.mean(opt, axis=0) * 1.507 * 180 / np.pi, np.max(opt, axis=0) * 1.507 * 180 / np.pi))
    hist = 0
    for i in range(len(opt)):
        if opt[i] < 3:
            hist += 1

    return output


# >================== パラメータ ================== <

# 入力関係の整備
Num_input_node = 1
Num_hidden_node = 80
Num_output_str = 1
max_length_of_seaquence = 15  # センサーの入力数

# 学習コンフィグの整理
training_epoch = 10000  # 学習会数
learning_rate = 0.0003  # デフォルトは0.01
forget_rate = 1.0  # LSTM忘却ゲート係数 推奨は1.0
mini_batch = 50  # バッチサイズ
num_of_prediction_epochs = 50  # 評価用のェポッチ数
num_of_prediction_data = num_of_prediction_epochs

LOG_DIR = './logs'
# >================== パラメータ(end) ================== <
# >================== 以下メイン処理 ================== <
# csvファイルの読み込み (後処理を見越してそれぞれ別名変数化)

# 問題用
data = []
L_pos = []
R_pos = []
yaw_ang = []
curv_R = []
velocity = []
tate_G = []
yoko_G = []
yaw = []
str_n = []
str_a = []
yoko_v = []
suberi = []
ttf05 = []
ttf08 = []
yoko_P = 0

# 解答用
L_pos_a = []
R_pos_a = []
yaw_ang_a = []
curv_R_a = []
velocity_a = []
tate_G_a = []
yoko_G_a = []
yaw_a = []
str_n_a = []
str_a_a = []
yoko_v_a = []
suberi_a = []
ttf05_a = []
ttf08_a = []
yoko_P_a = 0

# シミュレーションデータの取得
for num in range(1, 10):
    data = np.loadtxt('./RNN_LKA_TruckSim_N000%s.csv' % str(num), delimiter=',')

    L_pos = np.hstack((L_pos, data[:, 0]))
    R_pos = np.hstack((R_pos, data[:, 1]))
    yaw_ang = np.hstack((yaw_ang, data[:, 2]))
    curv_R = np.hstack((curv_R, data[:, 3]))
    velocity = np.hstack((velocity, data[:, 4]))
    tate_G = np.hstack((tate_G, data[:, 5]))
    yoko_G = np.hstack((yoko_G, data[:, 6]))
    yaw = np.hstack((yaw, data[:, 7]))
    str_n = np.hstack((str_n, data[:, 8]))
    # 9: 100f 10: 200f 11f:300 12f: 400
    str_a = np.hstack((str_a, data[:, 9]))
    yoko_v = np.hstack((yoko_v, data[:, 13]))
    suberi = np.hstack((yoko_v, data[:, 14]))
    ttf05 = np.hstack((ttf05, data[:, 15]))
    ttf08 = np.hstack((ttf08, data[:, 16]))
    print('%s done' % num)

# 解答用データの取得
for num in range(11, 19):
    data = np.loadtxt('./RNN_LKA_TruckSim_N000%s.csv' % str(num), delimiter=',')
    L_pos_a = data[:, 0]
    R_pos_a = data[:, 1]
    yaw_ang_a = data[:, 2]
    curv_R_a = data[:, 3]
    velocity_a = data[:, 4]
    tate_G_a = data[:, 5]
    yoko_G_a = data[:, 6]
    yaw_a = data[:, 7]
    str_n_a = data[:, 8]
    # 9: 100f 10: 200f 11f:300 12f: 400f
    str_a_a = data[:, 9]
    yoko_v_a = data[:, 13]
    suberi_a = data[:, 14]
    ttf05_a = data[:, 15]
    ttf08_a = data[:, 16]

    print('answer ready!')

# モデル呼び出し
with tf.Graph().as_default():
    preds = tf.placeholder(tf.float32, [None, max_length_of_seaquence, Num_input_node], name="input")
    Y = tf.placeholder(tf.float32, [None, Num_output_str], name="label")
    istate_ph = tf.placeholder(tf.float32, [None, Num_hidden_node * 2], name="istate")

    output_op, states_op, datas_op = inference(preds, istate_ph)
    # コストの定義
    cost = tf.reduce_mean(tf.square(Y - output_op) * 10000)
    tf.summary.scalar("loss", cost)

    # 確率的勾配降下法の定義
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    summary_op = tf.summary.merge_all()
    train_loss = tf.summary.scalar('loss', cost)
    X, t = create_data(num_of_samples=num_of_samples, input_sensors=max_length_of_seaquence)

    # 解答データの作成
    X_a, t_a = create_data_a(num_of_samples=test_sumples, input_sensors=max_length_of_seaquence)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        embedding_var = tf.Variable(tf.random_normal([mini_batch, 15], name='embedding'))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)
        projector.visualize_embeddings(summary_writer, config)

        # セッションの初期化
        sess.run(tf.global_variables_initializer())
        count = 1

        for epoch in range(training_epoch):
            batch_X, batch_Y, count = get_batch(mini_batch, X, t, count)
            # 初回処理
            if epoch == 0:
                train_dict = {
                    preds: batch_X,
                    Y: batch_Y,
                    istate_ph: np.zeros((mini_batch, Num_hidden_node * 2)),
                }
            # 次回度以降はstateを引き継ぐ
            else:
                train_dict = {
                    preds: batch_X,
                    Y: batch_Y,
                    istate_ph: state,
                }

            opt, state = sess.run([optimizer, states_op], feed_dict=train_dict)

            # accuracyの表示
            if epoch % 100 == 0:
                summary_str, train_loss = sess.run([summary_op, cost], feed_dict=train_dict)
                print("train#%d, train loss: %e" % (epoch, train_loss))
                summary_writer.add_summary(summary_str, epoch)
                if epoch % 500 == 0:
                    calc_accuracy(output_op, X_a, t_a, prints=True)
                    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))

        calc_accuracy(output_op, X_a, t_a, prints=True)
        datas = sess.run(datas_op)
        saver.save(sess, "./model_final.ckpt")
