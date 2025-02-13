text1="Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that, at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Leighton's work is really Greek after all, and can discover in it but little of rocky Ithaca."
text2="i swear answered sancho"

from evaluate import load

wer_metric = load("wer")
#wer_metric=load("eer")
wer = wer_metric.compute(references=[text1], predictions=[text2])

print(wer)