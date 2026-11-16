# Abstract

**Headline:** 6.70 GB -> 3.83 GB, with HarmBench refusal 79.2% -> 79.8% and English MMLU 56.2% -> 55.8%.

We partly recreate and extend two Lexsi Labs papers on Tiny Aya Global (3.35B), using all three Lexsi toolkits: SafeTune, CircuitKIT and AuditKIT. *Forgetting That Sticks* reports that 4-bit quantization can undo unlearning and proposes MANSU to prevent this. *C-ΔΘ* locates refusal behaviour in a sparse circuit found with EAP-IG. We ask two questions again:
- Does compression undo unlearning on Tiny Aya?
- Can a refusal circuit guide which layers keep full precision when the model is compressed?

**Unlearning.** We fine-tune Tiny Aya on TOFU and use SafeTune's NPO to forget 20 fictitious authors. Word-overlap retention on the forget set falls from 1.00 to 0.15, while English MMLU stays within 1.6 pp. We then compress with two 4-bit formats: quanto qint4 and bitsandbytes NF4. In both, the log-probability of the forgotten answers rises: by +0.215 nats/token [95% CI 0.129, 0.305] under qint4 and by +0.262 [0.196, 0.329] under NF4. That is 12–15% of what NPO removed. Compressing the fine-tuned model, which was never unlearned, lowers it instead. This is the direction the paper describes. Word overlap, our headline measure, and multiple-choice accuracy do not show it, so most of the unlearning survives 4-bit. Compression also costs the unlearned model more retained knowledge.

**Refusal.** CircuitKIT's EAP-IG selects 77 of 252 linear layers for refusal. We keep them in bf16 and quantize the rest to qint4, and we compare against plain qint4 and against 77 random layers. We evaluate on HarmBench, Hindi and Arabic Aya red-team prompts, XSTest and Global-MMLU, with WildGuard as the judge. All compressed models keep refusal within about 3 pp of the original, with or without protected layers. On our main comparison, HarmBench refusal against the random control, the protected model is similar (−2.4 pp [−6.0, 1.2]). Its advantage is fidelity: mean KL to the original is 15–31% lower than the random control's and 41–50% lower than plain qint4's on all four prompt sets, and every paired interval excludes zero. It also keeps MMLU best (−0.4 pp in English). Plain qint4 is 2.57 GB.

**MANSU.** We implement MANSU's forget circuit, Fisher-masked circuit training, NF4 magnitude floor and CAD diagnostics,. We confirm the paper's update-size gap: in its setting, training updates are 180–220 times smaller than one NF4 step. A random-sign control shows that a floor of this size is harmless on its own. On our strongly memorised TOFU model, training produced almost no forgetting signal, so the floored model broke our 8 pp utility limit and was not exported.

We release code, per-question numeric results, seven derived models and a CPU script that reproduces every table. The main limits are one seed, one random-layer draw, an automatic refusal judge and small Hindi and Arabic sets.
