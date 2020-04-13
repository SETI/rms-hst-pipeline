
import os
import os.path
from pdart.pds4.HstFilename import HstFilename

DOWNLOAD_DIR = '/Volumes/PDART-8TB/bulk-download'

def get_instr(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.fits':
                return HstFilename(file).instrument_name()

def get_size(dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def go():
    res = []
    for dir in os.listdir(DOWNLOAD_DIR):
        full_dir = os.path.join(DOWNLOAD_DIR, dir)
        instr = get_instr(full_dir)
        res.append( (instr, round(get_size(full_dir) / (1024 * 1024)), dir) )
    res.sort()
    for r in  res:
        print r

if __name__ == '__main__':
    go()
