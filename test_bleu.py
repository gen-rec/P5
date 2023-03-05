from notebooks.evaluate.utils import bleu_score
from nltk.translate.bleu_score import sentence_bleu

if __name__ == '__main__':
    ref = ["Hello world! A B C D E".split()]
    hyp = ["Hello world! A B E T Q".split()]

    print(bleu_score(ref, hyp, n_gram=4, smooth=False))
    print(sentence_bleu(ref, hyp[0], weights=(0.25, 0.25, 0.25, 0.25)) * 100)
