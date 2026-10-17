from pathlib import Path
import hashlib,json
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def save_json(path,value):
 Path(path).write_text(json.dumps(value,indent=2)+"\n")
