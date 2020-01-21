import os

from dglls.model import load_pretrained

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test():
    model = load_pretrained('DGMG_ChEMBL_canonical')
    model = load_pretrained('DGMG_ChEMBL_random')
    model = load_pretrained('DGMG_ZINC_canonical')
    model = load_pretrained('DGMG_ZINC_random')

    remove_file('DGMG_ChEMBL_canonical_pre_trained.pth')
    remove_file('DGMG_ChEMBL_random_pre_trained.pth')
    remove_file('DGMG_ZINC_canonical_pre_trained.pth')
    remove_file('DGMG_ZINC_random_pre_trained.pth')

if __name__ == '__main__':
    test()
