# LLM brain — captured run

A real run of the agent with the **LLM brain** (`--brain llm`) driving the
loop. The backends are mock, so the *scores* are synthetic — but the per-round
`[LLM]` **diagnoses are a live Anthropic model** reasoning over each round's
gate funnel and choosing how to steer the next round (a small `contact_bias`
nudge, a `theta` relaxation, a scaffold reseed, or stop). The model's answer is
returned as a forced tool call, clamped into safe ranges, and printed here.

Reproduce (needs your own Anthropic API key and a model your account can access):

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-ant-...
python -c "import anthropic; [print(m.id) for m in anthropic.Anthropic().models.list().data]"   # list your models
python -m pmhc_agent.run --brain llm --library 600 --model <model-id> | tee docs/llm-brain-run.md
```

Captured with `--model claude-sonnet-4-5`.

```text
==========================================================================
  pMHC-Design Agent  ·  campaign: MART-1 AAGIGILTV / HLA-A*02:01
  backends: ALL MOCK   seed=7   brain: llm
==========================================================================
  • Target ready: MART-1 AAGIGILTV on HLA-A*02:01 (%rank 0.651, struct AF3:460303).
  • Off-target panel: 12 confusable peptides (hardest: ['AAGLGILTV', 'AAGTGILTV', 'AAGIDILTV']).
--------------------------------------------------------------------------
  Round 0:  generated 480 designs
      G2 peptide-centric         pass  302  reject   178
      G3 foldable                pass  230  reject    72
      G4 fold&dock               pass  167  reject    63
      G5 contrastive recovery    pass  167  reject     0
      G6 specificity margin      pass  160  reject     7  (theta=1.40)
      G7 interface partition     pass  151  reject     9
      G8 consensus               pass  144  reject     7
      G9 developable             pass  144  reject     0
      -> survivors this round: 144   |  library so far: 144
      diagnosis: [LLM] Round 0 is healthy: G2 attrition (178/480) is moderate and later gates (G5-G9) pass at high rates with only minor G6 losses, indicating designs are largely on-target and specific. No systemic bottleneck warrants aggressive intervention; a light contact_bias nudge can further reduce G2/G4 losses without disrupting a working pipeline.

  Round 1:  generated 480 designs
      G2 peptide-centric         pass  362  reject   118
      G3 foldable                pass  283  reject    79
      G4 fold&dock               pass  217  reject    66
      G5 contrastive recovery    pass  217  reject     0
      G6 specificity margin      pass  178  reject    39  (theta=1.64)
      G7 interface partition     pass  178  reject     0
      G8 consensus               pass  169  reject     9
      G9 developable             pass  169  reject     0
      -> survivors this round: 169   |  library so far: 313
      diagnosis: [LLM] Round 1 is healthy: 169/480 survive with no stalling, G5 fully passes (designs recognize target) and only modest attrition at G2/G4/G6. The main soft bottleneck is G6 (39 rejected just below margin), suggesting a small theta relaxation plus mild contact bias would improve yield without masking cross-reactivity.

  Round 2:  generated 480 designs
      G2 peptide-centric         pass  434  reject    46
      G3 foldable                pass  341  reject    93
      G4 fold&dock               pass  266  reject    75
      G5 contrastive recovery    pass  266  reject     0
      G6 specificity margin      pass  193  reject    73  (theta=1.74)
      G7 interface partition     pass  193  reject     0
      G8 consensus               pass  186  reject     7
      G9 developable             pass  186  reject     0
      -> survivors this round: 186   |  library so far: 499
      diagnosis: [LLM] Most designs pass G5 (recognize target) but a notable fraction (73/266) fail the G6 specificity margin, indicating designs are close but not sharply discriminating; G2/G4 attrition is moderate and not the dominant bottleneck. Small theta relaxation plus a scaffold reseed should recover near-miss designs without sacrificing specificity, and stalled_rounds=0 means no need to escalate.

  Round 3:  generated 480 designs
      G2 peptide-centric         pass  424  reject    56
      G3 foldable                pass  354  reject    70
      G4 fold&dock               pass  285  reject    69
      G5 contrastive recovery    pass  285  reject     0
      G6 specificity margin      pass  186  reject    99  (theta=1.77)
      G7 interface partition     pass  186  reject     0
      G8 consensus               pass  175  reject    11
      G9 developable             pass  175  reject     0
      -> survivors this round: 175   |  library so far: 674
      diagnosis: Target library size reached.

--------------------------------------------------------------------------
  Final stage: library   | accepted designs: 674   | adaptive theta: 1.768
==========================================================================
  RANKED SHORTLIST (human-approved synthesis required):

    design           composite   margin   pAE_i   pLDDT    phi  worst_off
    r2_bb211_s0         0.7734     2.37     3.0    89.8   0.76  AAGLGILTV
    r3_bb8_s1           0.7697     2.15     4.2    93.4   0.79  AAGIFILTV
    r2_bb106_s1         0.7660     2.17     3.1    92.9   0.77  AAGLGILTV
    r3_bb46_s1          0.7550     2.52     3.1    93.1   0.80  AAGIFILTV
    r3_bb179_s0         0.7521     2.18     3.9    90.9   0.79  AAGIFILTV
    r2_bb89_s1          0.7472     2.13     4.2    90.1   0.79  AAGIGILKV
    r1_bb168_s1         0.7433     2.28     3.0    92.1   0.72  AAGIFILTV
    r2_bb212_s0         0.7384     2.27     3.0    91.6   0.76  AAGIGCLTV
    r3_bb60_s0          0.7383     2.40     3.5    91.9   0.76  AAGIGCLTV
    r3_bb7_s1           0.7312     2.29     4.3    91.4   0.79  AAGTGILTV

  Reminder: DNA ordering, assays, and spend are HUMAN-GATED.
  The agent recommends; a scientist commits.
```
