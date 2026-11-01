"""SenseNova smoke test: batched greedy decoding with row dropping, native1 / native2 / splice21."""
import sys, time
from ouro.sensenova_runner import SenseNovaRunner, tokenizer
from ouro.evaluate_bench import greedy_batch

path = sys.argv[1]
tok = tokenizer(path)
tok.stop_ids = {248044, 248046, tok.eos_token_id}
qs = ["What is 17 * 23? Answer with the number only.", "Name the capital of France in one word.",
      "Janet has 3 apples and buys 5 more, then eats 2. How many apples does she have? Think step by step."]
rows = [dict(prompt_ids=tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True,
                                                 enable_thinking=False) if False else tok(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False))["input_ids"],
             until=[], max_new_tokens=n) for q, n in zip(qs, (16, 8, 200))]
for cond in sys.argv[2:]:
    r = SenseNovaRunner(path, cond)
    t = time.time()
    batched = greedy_batch(r, tok, rows)
    single = [greedy_batch(r, tok, [x])[0] for x in rows]
    print(cond, f"{time.time() - t:.1f}s")
    for b, s in zip(batched, single):
        print("  same" if b[0] == s[0] else "  DIFF", repr(b[0][-160:]), b[1], b[2])
    del r
