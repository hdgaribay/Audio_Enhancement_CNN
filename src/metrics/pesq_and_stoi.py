
from pesq import pesq
from pystoi import stoi

def compute_pesq(clean_wav, enhanced_wav):
    return pesq(16000, clean_wav, enhanced_wav, "wb")

def compute_stoi(clean_wav, enhanced_wav):
    return stoi(clean_wav, enhanced_wav, 16000, extended=False)