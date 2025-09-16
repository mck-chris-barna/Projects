What this model does

Detects the overall sentiment (emotional tone) of a short piece of text.

Classifies text into one of three categories:

Negative

Neutral

Positive

Works by spotting statistical patterns in training data, not by "understanding" language.

What the results look like

Example output for one sentence:

Negative: 0.72

Neutral: 0.23

Positive: 0.05

How to read this:

Negative = 0.72 → 72% confident the text is negative.

Neutral = 0.23 → 23% confident it’s neutral.

Positive = 0.05 → 5% confident it’s positive.

The highest score = predicted sentiment (in this case: Negative).

How to interpret the scores

High score (≥ 0.70): model is confident in its prediction.

Scores close together: the text is ambiguous or mixed in tone.

High Neutral score: the text lacks strong emotion (plain or factual).

Examples

“I love this new update!” → Positive (score ~0.90).

“This app is fine, nothing special.” → Neutral (score ~0.70).

“Worst service ever.” → Negative (score ~0.95).

Important notes

Model was trained on tweets, so it performs best on short, casual text.

Sarcasm/slang/jokes may cause errors. Example: “Great, another bug” might be misread as Positive.

Close scores across categories = uncertainty. Treat with caution.

Summary

Look for the category with the highest score → that’s the predicted sentiment.

The score itself = confidence level.

Higher score = more certainty.

Close scores = ambiguous, may need manual review.
