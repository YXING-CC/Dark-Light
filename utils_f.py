import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1)

def classification_results(pred, lab):
    # print('In utils pred.size: ', pred[1].size(), pred[1], 'lab.size: ', lab.size(), lab)
    dim = len(pred)
    runing_corrects = torch.zeros(dim).to(device)

    # for i in range(dim):
    #     cylind_test = softmax(pred[i])
    #
    #     preds_test = torch.max(cylind_test, 1)[1]
    #
    #     runing_corrects[i] = torch.sum(preds_test == lab[:, i])

    cylind_test_0 = softmax(pred[0])
    preds_test_0 = torch.max(cylind_test_0, 1)[1]
    runing_corrects[0] = torch.sum(preds_test_0 == lab[:, 0])

    cylind_test_1 = softmax(pred[1])
    preds_test_1 = torch.max(cylind_test_1, 1)[1]
    runing_corrects[1] = torch.sum(preds_test_1 == lab[:, 1])

    cylind_test_2 = softmax(pred[2])
    preds_test_2 = torch.max(cylind_test_2, 1)[1]
    runing_corrects[2] = torch.sum(preds_test_2 == lab[:, 2])

    cylind_test_3 = softmax(pred[3])
    preds_test_3 = torch.max(cylind_test_3, 1)[1]
    runing_corrects[3] = torch.sum(preds_test_3 == lab[:, 3])

    pred_test = [preds_test_0, preds_test_1, preds_test_2, preds_test_3]
    lab_test = [lab[:, 0], lab[:, 1], lab[:, 2], lab[:, 3]]

    return runing_corrects, pred_test, lab_test


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe




