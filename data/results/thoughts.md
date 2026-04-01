# Day 1:

## I ran this code below:

```
probabilities_singular = get_verb_probabilities(model, tokenizer, "The key to the cabinet", "is", "are")
probabilities_plural = get_verb_probabilities(model, tokenizer, "The key to the cabinets", "is", "are")
print("Singular:", probabilities_singular)
print("Plural:", probabilities_plural)
```

## Got this result:

```
Singular: {'is': 0.12984387576580048, 'are': 0.0020903670229017735}
Plural: {'is': 0.34607890248298645, 'are': 0.058304376900196075}
```

## Analysis:

| Condition | P("is") | P("are") | is/are ratio |
|-----------|---------|----------|--------------|
| Baseline ("cabinet") | 0.130 | 0.002 | ~62x |
| Attractor ("cabinets") | 0.346 | 0.058 | ~6x |

Model gets it right both times but confidence drops around 10x when the plural attractor is present. P("are") jumps 28x. GPT-2 Small is partially attracted but resists so the next question is what circuit enables that resistance. **Next step:** run across 15-20 sentence pairs to confirm the effect is consistent.




