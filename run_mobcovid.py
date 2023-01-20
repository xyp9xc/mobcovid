import sys, os

if not 'mobcovid' in sys.path:
    sys.path += ['mobcovid']

from utils.tools import dotdict
from exp.exp_mobcovid import Exp_Informer
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
import seaborn as sns
import numpy as np

def main_fun(seq_len1, seq_len2, label_len):
    args = dotdict()

    args.model = 'mobcovid' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'LPD' # Latetime Population Data
    args.root_path = './' # root path of data file
    args.data_path = 'Osaka.csv'#'normalized_data_final.csv' # data file
    # args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = ['latetime population', 'confirmed_each_day'] # target feature in S or MS task
    args.freq = 'd' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = './informer_checkpoints' # location of model checkpoints

    args.seq_len1 = seq_len1 # input sequence length of latetime population
    args.seq_len2 = seq_len2 # input sequence length of confirmed cases
    args.label_len = label_len # start token length of Informer decoder
    if args.data_path == "Tokyo.csv":
        args.emergency_in = 2
    elif args.data_path == "Osaka.csv":
        args.emergency_in = 2
    args.pred_len = 1 # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.loss_target_1_weight = 1#0.005
    args.loss_target_2_weight = 1# - args.loss_target_1_weight

    args.enc_in = 1 # encoder input size ------embed------
    args.dec_in = 1 # decoder input size


    args.c_out = 1 # output size

    args.factor = 5 # probsparse attn factor
    args.d_model = 256 # dimension of model
    args.n_heads = 8 # num of heads

    args.e_layers = 1 # num of encoder layers
    args.d_layers = 1 # num of decoder layers

    args.d_ff = 512 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu' # activation
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in encoder
    args.mix = True
    args.padding = 0

    args.batch_size = 8
    args.learning_rate = 0.001
    args.loss = 'mae'
    args.lradj = 'type2'
    args.use_amp = False # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 10000
    args.patience = 10
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    args.inverse = False #change


    seed = 7
    torch.cuda.manual_seed(seed)
    seed0 = 7
    torch.manual_seed(seed0)
    seed1 = 7
    np.random.seed(seed1)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Informer
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_param_test_{}_{}_sl1{}_sl2{}_ll{}_pl{}'.format(
            args.target,
            args.model,
            args.data_path.split(".")[0],
            args.seq_len1,
            args.seq_len2,
            args.label_len,
            args.pred_len,
)

        # set experiments
        exp = Exp(args)

        # 训练集：测试集：验证集 = 6:2:2

        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse_1, mae_1, mape_1, mse_2, mae_2, mape_2 = exp.test(setting)

        # predict
        # print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.predict(setting)

        torch.cuda.empty_cache()

    # set saved model path
    # setting = 'informer_LPD_ftS_sl60_ll30_pl15_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
    # setting = 'informer_LPD_ftS_sl60_ll30_pl15_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
    # path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

    # exp = Exp(args)
    #
    #
    # print(exp.predict(setting, True))
    #
    # prediction = np.load('./results/'+setting+'/real_prediction.npy')
    #
    # print(prediction.shape)




    preds_1 = np.load('./results/'+setting+'/pred_latetime population_rescale.npy')
    trues_1 = np.load('./results/'+setting+'/true_latetime population_rescale.npy')
    preds_2 = np.load('./results/'+setting+'/pred_confirmed_each_day_rescale.npy')
    trues_2 = np.load('./results/' + setting + '/true_confirmed_each_day_rescale.npy')
    # true = []
    # pred = []
    #
    # for i in range(trues.shape[0]):
    #     true.append(np.mean(trues[i, -1, -1]))
    #     pred.append(np.mean(preds[i, -1, -1]))
    # for i in range(len(pred)):
    #     print("{:.3}, {:.3}".format(pred[i], true[i]))
    plt.figure(figsize=(12, 8))
    plt.plot(trues_1, label='GroundTruth')
    plt.plot(preds_1, label='Prediction')
    plt.legend()
    plt.xlabel("day")
    plt.ylabel("Nighttime staying people")
    if args.data_path == 'Osaka.csv':
        plt.ylim([500000, 750000])
    # plt.ylim([0, 1])
    plt.savefig('./results/' + setting + '/figure_latetime population_rescale.png')
    # plt.show()
    plt.figure(figsize=(12, 8))
    plt.plot(trues_2, label='GroundTruth')
    plt.plot(preds_2, label='Prediction')
    plt.legend()
    plt.xlabel("day")
    plt.ylabel("Confirmed cases")
    if args.data_path == 'Osaka.csv':
        plt.ylim([0, 1500])
    else:
        plt.ylim([0, 1200])
    plt.savefig('./results/' + setting + '/figure_confirmed_each_day_rescale.png')

    return mse_1, mae_1, mape_1, mse_2, mae_2, mape_2

if __name__=="__main__":
    # error_list = []
    # for i in range(2, 22):
    #     for j in range(2, 22):
    #         mse_1, mae_1, mape_1, mse_2, mae_2, mape_2 = main_fun(seq_len1=i, seq_len2=j, label_len=1)
    #         error_list.append((i, j, mse_1, mae_1, mape_1, mse_2, mae_2, mape_2))
    # for k in range(len(error_list)):
    #     print("latetime population", error_list[k][0], "confirmed cases", error_list[k][1])
    #     print("latetime population mse:{}, mae:{}, mape:{}".format(error_list[k][2], error_list[k][3], error_list[k][4]))
    #     print("confirmed cases mse:{}, mae:{}, mape:{}".format(error_list[k][5], error_list[k][6], error_list[k][7]))
    #     print("")
    main_fun(seq_len1=5, seq_len2=3, label_len=1)
