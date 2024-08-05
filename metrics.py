import torch
from torch_scatter import scatter_add


def cal_gini(predict, item_count, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)

    def cal_gini_coefficient(index):
        import numpy as np
        index = index.reshape(-1)
        output = torch.zeros([item_count], device=index.device).long()
        output = scatter_add(torch.ones_like(index), index=index.long(), out=output).tolist()
        sorted_x = np.sort(output)
        n = len(output)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    gini = [round(cal_gini_coefficient(topk_predict[:, :ks[i]]), 4) for i in range(len(ks))]
    return gini


def cal_cratio(predict, item2pop, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    item2pop = torch.tensor(item2pop)

    item2pop[0] = 0
    threshold_pop = sorted(item2pop)[int(len(item2pop) * 0.8)]
    tail_item = torch.tensor([1 if item < threshold_pop else 0 for item in item2pop])

    def get_cratio(item_matrix):
        item_matrix = item_matrix.reshape(-1)
        tail_matrix = tail_item[item_matrix]
        return 1 - tail_matrix.sum() / len(tail_matrix)
    c_ratio = [round(get_cratio(topk_predict[:, :ks[i]]).item(), 4) for i in range(len(ks))]
    return c_ratio


def cal_recall(label, predict, ks):
    label = label.unsqueeze(-1)
    predict = predict.cpu().float()
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [round(hit[:, :ks[i]].sum().item()/label.size()[0], 4) for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    label = label.unsqueeze(-1)
    predict = predict.cpu().float()
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append(round((predict_dcg/max_dcg).mean().item(), 4))
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel

