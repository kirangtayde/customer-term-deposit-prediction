import numpy as np
from src.evaluate import classification_report_summary, threshold_table

def test_evaluation_metrics():
    y=np.array([0,1,1,0,1,0])
    p=np.array([.1,.8,.7,.2,.9,.3])
    m=classification_report_summary(y,p,.5)
    assert 0 <= m['roc_auc'] <= 1
    assert set(['precision','recall','f1']).issubset(m)
    assert len(threshold_table(y,p)) == 7
