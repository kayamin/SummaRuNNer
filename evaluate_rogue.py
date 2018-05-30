from sumeval.metrics.rouge import RougeCalculator

with open('./outputs/ref/1.txt') as ref:
    reference = ref.read()
    reference = '. '.join(reference.split('\n'))

with open('./outputs/hyp/1.txt') as hyp:
    summary = hyp.read()
    summary = '. '.join(summary.split('\n'))

rouge = RougeCalculator(stopwords=True, lang="en")

rouge2 = rouge.rouge_n(
    summary=summary,
    references=reference,
    n=2
)

print(rouge2)

