# formwork

**Declare which fields your code owns and which the model may invent, then get an object that satisfies your domain rules — or an exception. Never something in between.**

```
pip install formwork
```

> Status: **v0.1, alpha.** The core loop, its tests and the Gemini adapter are real. The benchmark is not, so the efficiency claims below are reasoned rather than measured. See [Roadmap](#roadmap).

---

## The problem

Grammar-constrained decoding solved one half of structured output. Outlines, XGrammar and friends will guarantee that the model returns valid JSON matching your schema — that `sets` is an integer between 1 and 10.

They cannot guarantee the other half:

- `exercise_id` must exist in the 200-row library you loaded three lines ago
- weekly sets per muscle must land in the range your programming logic decided on
- the meal's macros must sum to today's target within 5%
- no exercise may load a joint listed in the user's injury notes

These are **relational and context-dependent**. They are not expressible in a grammar, because they depend on runtime state and arithmetic across fields. So everyone writes the same thing by hand: validate, stuff the errors into a string, regenerate the whole object, hope.

That hand-rolled loop has three problems. It throws away the 90% of the output that was correct. It gives the model fresh opportunities to break a rule it had already satisfied. And it is nearly impossible to test, because testing it means calling an LLM.

## What formwork does

One class carries the schema, the ownership split, and the rules:

```python
from typing import Annotated
from pydantic import Field
from formwork import Spec, computed, chosen, generated, rule, generate
from formwork import repair

class WorkoutPlan(Spec):
    """A week of training for one person."""

    # 1. Your rule engine owns these. The model never sees them as a choice —
    #    they are stated as facts it must write around.
    split:      Annotated[str,            computed(rules.split_for)]
    set_range:  Annotated[tuple[int, int], computed(rules.weekly_sets)]

    # 2. The model picks, but only from a closed set you supply at runtime.
    exercises:  Annotated[list[Exercise], chosen(source="library", key="id")]

    # 3. The model is free.
    rationale:  Annotated[str, generated(describe="Two sentences, plain language.")]

    @rule("Every exercise must come from the supplied library.",
          fields=["exercises"],
          repair=repair.drop_invalid("exercises"))
    def known_exercises(self, ctx):
        missing = [e.id for e in self.exercises if e.id not in ctx.library_ids]
        return [f"{m} is not in the library" for m in missing]

    @rule("Weekly sets per muscle must stay inside the prescribed range.",
          fields=["exercises"])
    def volume(self, ctx):
        low, high = self.set_range
        for muscle, total in self.sets_by_muscle().items():
            if not low <= total <= high:
                yield f"{muscle}: {total} sets, allowed {low}-{high}"

plan, report = generate(WorkoutPlan, ctx, model)
```

`plan` satisfies every rule. If it could not, you got a `ConstraintError` instead — never a quietly invalid plan.

### The three roles

| | who fills it | in the model's schema? |
|---|---|---|
| `computed(fn)` | your code, before the model runs | **no** |
| `chosen(source=...)` | the model, from a closed set | yes |
| `generated()` | the model, freely | yes |

The schema handed to the model contains only the last two. **A field the model cannot see is a field it cannot get wrong** — and this single split removes more failures than any amount of prompt tuning, because most "hallucinations" in these features are the model answering a question you never needed to ask it.

### Repair, not retry

When a rule fails, formwork does not regenerate the object.

1. If the rule declared a `repair=`, it runs. `drop_invalid`, `truncate`, `clamp`, `dedupe` and `chain` ship with the library. No model call.
2. Otherwise it builds a **targeted repair**: the fields the violated rules named, and only those, are re-requested. Everything else is frozen and kept.
3. Out of attempts → `ConstraintError` carrying the surviving violations and the full cost.

```python
report.summary()
# '2 model call(s); 1,430 tokens; repaired by known_exercises'
report.attempts[1].targeted_fields
# ('exercises',)
```

### Soft constraints

Not everything is pass/fail. `@soft` declares a score to minimise, and `candidates=N` keeps the rule-valid result that scores best:

```python
@soft(weight=1.0)
def volume_balance(self, ctx):
    return stdev(self.sets_by_muscle().values())

plan, report = generate(WorkoutPlan, ctx, model, candidates=3)
```

## Testing, without a network

This is the part worth stealing even if you use none of the rest.

```python
from formwork.providers import Chaos, Recording

def test_never_emits_an_invalid_plan():
    for seed in range(200):
        model = Chaos(baseline=VALID_RESPONSE, seed=seed)
        try:
            plan, _ = generate(WorkoutPlan, ctx, model, max_attempts=3)
        except ConstraintError:
            continue           # giving up is allowed
        assert plan.check(ctx).ok   # lying is not
```

`Chaos` returns plausible-but-wrong variations of a baseline: unknown ids, counts over the limit, numbers out of range, duplicated entries. That is the failure mode that actually bites — not "the model returned garbage", but "the model returned something that looked fine".

`Recording` wraps any model so you can assert on the loop itself, which is how you keep the efficiency claim honest:

```python
model = Recording(Scripted([bad_plan, good_exercises]))
generate(WorkoutPlan, ctx, model)
assert model.requested_fields == [
    ("exercises", "rationale"),   # first call: everything the model owns
    ("exercises",),               # repair: only what broke
]
```

## Bring your own model

One adapter ships today:

```bash
pip install "formwork[gemini]"
```

```python
from formwork.providers.gemini import Gemini

plan, report = generate(WorkoutPlan, ctx, Gemini())   # reads GEMINI_API_KEY
```

It converts each schema into the OpenAPI subset Gemini's `response_schema`
accepts — inlining `$ref`s and dropping `additionalProperties`, both of which
are 400s otherwise. Anything it drops is still enforced by the engine when the
response comes back.

For anything else, the interface is one method. Adapting a provider is a
twenty-line job:

```python
class MyModel:
    def generate_structured(self, request) -> tuple[dict, Usage]:
        response = client.responses.parse(
            model="...",
            instructions=request.system,
            input=request.prompt,
            text_format=request.schema,        # a Pydantic model
        )
        return response.output_parsed.model_dump(), Usage(...)
```

Async works the same way via `agenerate`. Both drivers are thin wrappers over a sans-IO `Session` state machine, so if you need to drive the loop yourself — a queue, a batch API, a human in the loop — you can:

```python
session = Session(WorkoutPlan, ctx)
while (request := session.next_request()) is not None:
    raw, usage = however_you_like(request)
    session.feed(raw, usage)
plan, report = session.finish()
```

## How this differs from what you may already use

| | what it constrains |
|---|---|
| **Outlines / XGrammar / llguidance** | output **format**, at decode time. Composes with formwork rather than competing — grammar for shape, formwork for meaning. |
| **Instructor / PydanticAI** | schema validation plus a retry. Closest neighbour. The differences: no notion of fields your code owns, validation is pass/fail with no soft scores, and the retry regenerates everything instead of the fields that broke. |
| **OR-Tools / python-constraint** | can express the constraints, cannot write the prose or make the taste judgement you wanted a model for. |
| **formwork** | **semantic constraints over runtime context**, with ownership declared per field and repair targeted at the failure. |

## Roadmap

- [x] Field roles, rules, deterministic + targeted repair, sans-IO session, test doubles
- [x] Gemini adapter, verified live — including that the provider accepts the
      narrowed schema a targeted repair generates on the fly
- [ ] Further provider adapters (Anthropic, OpenAI, Ollama)
- [ ] Grammar backend for `chosen` fields — enforce closed sets at decode time instead of validating after
- [ ] **Benchmark**: valid-on-first-try rate and tokens-to-convergence against a plain-prompt and an Instructor baseline, across three domains. Until this exists, treat the efficiency claims in this README as untested.
- [ ] `formwork.contrib` with worked specs for scheduling, meal planning and budgeting

## Why it exists

It was extracted from a fitness app where the same loop — deterministic rule engine, closed exercise library, semantic validation, one retry with feedback — had been written four times by hand, in four slightly different ways. The fourth time is when you write the library.

## License

MIT
