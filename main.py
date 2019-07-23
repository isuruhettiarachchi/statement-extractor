import textrank
import sys

path_stage0 = "input.json"
path_stage1 = "o1.json"

with open(path_stage1, 'w') as f:
    for graf in textrank.parse_doc(textrank.json_iter(path_stage0)):
        f.write("%s\n" % textrank.pretty_print(graf._asdict()))
        # to view output in this notebook
        print(textrank.pretty_print(graf))

path_stage1 = "o1.json"
path_stage2 = "o2.json"

graph, ranks = textrank.text_rank(path_stage1)
textrank.render_ranks(graph, ranks)

with open(path_stage2, 'w') as f:
    for rl in textrank.normalize_key_phrases(path_stage1, ranks):
        f.write("%s\n" % textrank.pretty_print(rl._asdict()))
        # to view output in this notebook
        print(textrank.pretty_print(rl))


path_stage1 = "o1.json"
path_stage2 = "o2.json"
path_stage3 = "o3.json"

kernel = textrank.rank_kernel(path_stage2)

with open(path_stage3, 'w') as f:
    for s in textrank.top_sentences(kernel, path_stage1):
        f.write(textrank.pretty_print(s._asdict()))
        f.write("\n")
        # to view output in this notebook
        print(textrank.pretty_print(s._asdict()))


path_stage2 = "o2.json"
path_stage3 = "o3.json"

phrases = ", ".join(set([p for p in textrank.limit_keyphrases(path_stage2, phrase_limit=12)]))
sent_iter = sorted(textrank.limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
s = []

for sent_text, idx in sent_iter:
    s.append(textrank.make_sentence(sent_text))

graf_text = " ".join(s)
print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))
