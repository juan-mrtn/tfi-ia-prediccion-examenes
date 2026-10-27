"""Barrido de rho para ART 1 con codificación posicional (ISSUE-103)."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.preprocessor import DataPreprocessor
from src.em_clustering import EMClusterer, academic_weights
from src.encoder import PositionalBinaryEncoder
from src.art1 import CarpenterGrossbergNetwork
from src.predictor import ExamPredictor, score_to_category
p=DataPreprocessor('data/metadata/student_metadata.csv')
raw=p.validate_hourly_consistency(p.load('data/raw/student_performance.csv'))
tr,te=p.train_test_split(raw)
em=EMClusterer(n_components=4).fit(tr[p.feature_cols].values.astype(float))
cats=em.compute_spectral_risk(academic_weights(p.feature_cols))
em_tr=np.array([cats[l] for l in em.predict(tr[p.feature_cols].values.astype(float))[0]])
str_,ste=p.scale_to_base_100_int(tr),p.scale_to_base_100_int(te)
binning=p.metadata_df['binning'].to_dict()
y_tr=tr.exam_score.values/10; y_te=te.exam_score.values/10
real_te=np.array([score_to_category(v) for v in y_te])
rows=[]
for code in ['posicional']:
  enc=PositionalBinaryEncoder(binning)
  Xtr,_=enc.fit_transform(str_,p.feature_cols); Xte,_=enc.fit_transform(ste,p.feature_cols)
  for rho in np.round(np.arange(0.10,0.91,0.05),2):
    t0=time.time()
    net=CarpenterGrossbergNetwork(Xtr.shape[1],max_categories=3000,rho=rho)
    try: lab=net.fit_predict(Xtr)
    except RuntimeError: rows.append(dict(rho=rho,clusters='>3000'));continue
    m=lab>=0
    pur=pd.DataFrame({'n':lab[m],'c':em_tr[m]}).groupby('n')['c'].agg(lambda s:s.value_counts().iloc[0]).sum()/m.sum()
    pr=ExamPredictor(net).calibrate(lab,em_tr,y_tr).predict(Xte)
    ok=pr.nodo_art1.values>=0
    rows.append(dict(rho=rho,clusters=net.num_active_clusters,CR=round(net.num_active_clusters/len(Xtr),4),
      RR=round(net.total_resets/len(Xtr),2),vacios_train=int((Xtr.sum(1)==0).sum()),pureza_EM=round(pur,3),
      MAE=round(np.abs(pr.nota_esperada.values-y_te).mean(),3),
      acc_cat=round((pr.categoria_pronosticada.values==real_te).mean(),3),
      conf_baja=round(pr.confianza_baja.mean(),3),seg=round(time.time()-t0,1)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('data/output/barrido_rho_art1.csv',index=False)
# baseline: predict EM-class mean directly
lab_te=em.predict(te[p.feature_cols].values.astype(float))[0]; lab_tr=em.predict(tr[p.feature_cols].values.astype(float))[0]
mm={k:y_tr[lab_tr==k].mean() for k in range(4)}; yh=np.array([mm[k] for k in lab_te])
print('EM solo MAE',round(np.abs(yh-y_te).mean(),3),'acc',round((np.array([cats[k] for k in lab_te])==real_te).mean(),3), 'media global MAE',round(np.abs(y_tr.mean()-y_te).mean(),3))
