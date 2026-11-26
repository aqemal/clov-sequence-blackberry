import io,json,zipfile,unittest
import pandas as pd
from dataset_release import ingest_report_archive,release_bundle

def archive(items):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for name,obj in items:
   z.writestr(name,json.dumps(obj) if isinstance(obj,dict) else obj)
 return b.getvalue()

def labels(n=10):
 return pd.DataFrame([dict(sample_id=f's{i}',label=('malicious' if i%2 else 'benign'),malware_family=('BankBot' if i%2 else 'benign'),apk_sha256=f'{i+1:064x}',label_evidence='test fixture') for i in range(n)])

def reports(n=10):
 return [(f'{i}.json',dict(sample_id=f's{i}',apk_sha256=f'{i+1:064x}',events=[f'app_start_{i}',f'call_{i}'])) for i in range(n)]

class TestRelease(unittest.TestCase):
 def test_ingest(self):
  g,b=ingest_report_archive(archive(reports()),labels())
  self.assertEqual(len(g),10);self.assertTrue(b.empty)
  self.assertEqual(g.iloc[0]['label'],'benign')
 def test_hash_mismatch(self):
  rr=reports(2);rr[0][1]['apk_sha256']='0'*64
  g,b=ingest_report_archive(archive(rr),labels(2))
  self.assertEqual(len(g),1);self.assertEqual(len(b),1)
 def test_no_family_inference(self):
  m=labels(1);m.loc[0,'label']='unknown'
  with self.assertRaises(ValueError):ingest_report_archive(archive(reports(1)),m)
 def test_reject_apk(self):
  g,b=ingest_report_archive(archive(reports(1)+[('test.apk',b'not-an-apk')]),labels(1))
  self.assertEqual(len(g),1);self.assertEqual(len(b),1)
 def test_duplicate_report(self):
  g,b=ingest_report_archive(archive(reports(1)+[('copy.json',reports(1)[0][1])]),labels(1))
  self.assertEqual(len(g),1);self.assertEqual(len(b),1)
 def test_release(self):
  g,b=ingest_report_archive(archive(reports(20)),labels(20))
  data,meta=release_bundle(g)
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   self.assertIn('train.csv',z.namelist());self.assertIn('test.csv',z.namelist())
   tr=pd.read_csv(z.open('train.csv'));te=pd.read_csv(z.open('test.csv'))
   self.assertFalse(set(tr.sample_id)&set(te.sample_id))
   self.assertEqual(set(tr.label),{'benign','malicious'});self.assertEqual(set(te.label),{'benign','malicious'})
 def test_duplicate_hash_blocks_release(self):
  g,b=ingest_report_archive(archive(reports(20)),labels(20))
  g.loc[1,'apk_sha256']=g.loc[0,'apk_sha256']
  with self.assertRaisesRegex(ValueError,'Repeated APK'):release_bundle(g)

if __name__=='__main__':unittest.main()
