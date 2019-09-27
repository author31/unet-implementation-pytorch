#metric
def dice(preds,targs):
  preds = (preds>0).float()
  return 2.* (preds*targs).sum()/(preds+targs).sum()