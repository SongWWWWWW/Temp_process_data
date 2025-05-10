# /bin/bash
echo $1
echo $2
echo "start download package"
pip install -r requirements.txt

echo "start download file"
python download.py --start $1   --end $2


echo "start transform to jsonl"
python data2jsonl.py


echo "start analysis"
python analysis.py

echo "upload dir"
python upload_dir.py

echo "upload file"
python upload.py
