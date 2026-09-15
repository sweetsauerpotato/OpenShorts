# Clip picking rules

The agent that chooses clips reads this with every transcript (MCP
`get_transcript`). Plain instructions: edit freely, it is read fresh on each
call. The creator's instructions for a single video come on top of these.

## Who watches

- Gen Z on TikTok, Instagram Reels and YouTube Shorts: scrolling fast, deciding
  in 1-2 seconds, often with the sound off (the captions carry it).
- A clip earns its place when a stranger stops, watches to the end, and wants
  to comment, share or rewatch. "Interesting" or "informative" is not enough.

## A moment is a clip when it has

1. A first line that stops the scroll on its own: a bold claim, a confession,
   a surprising number, a question you need answered, a fight starting.
2. A payoff inside the clip: the punchline, the reveal, the verdict, the
   answer, a strong opinion said plainly.
3. At least one of: real emotion (laughing, anger, awkwardness, disbelief,
   vulnerability); something relatable (money, dating, school, work, family,
   status, mental health); a take people will argue about; something useful
   they can use today.
4. No context needed: it makes sense to someone who saw nothing else.

Specific beats general: names, numbers and concrete details over abstractions.

## Never

- Intros, outros, "like and subscribe", sponsor reads, ads, housekeeping,
  "as I said earlier".
- A setup whose payoff is outside the clip, a list cut off halfway, an inside joke.
- A moment that needs the picture to make sense: the transcript cannot show
  what is on screen.
- A quote cut so it says something the speaker did not mean.
- Two clips that make the same point.
- Padding the count: fewer great clips beat more average ones.

## Cutting (a clip can be several pieces of the video, in any order)

- Total 20-45 s. Up to 60 s only if it holds tension all the way; never under 15 s.
- Open on the strongest line. When it comes late, lift it to the front as a
  cold open (2-6 s), then play the setup that leads to it. Don't play it twice
  unless the repeat is the payoff.
- Cut what drags: pauses, filler ("um", "you know", "like", "basically"),
  false starts, tangents, repeated sentences.
- Read a stretch word by word (`get_transcript` with `words=true`) before
  cutting inside a sentence. Cut between words, at a breath; never drop a word
  the meaning needs.
- End on the payoff or a punchy last line: never mid-thought, never on
  "so yeah", never on a question answered later.
- At most 12 pieces in a clip, none shorter than half a second.

## Hook (burned on screen for the first 5 seconds)

- At most 10 words, specific to this clip: name what it reveals or the tension
  it opens ("He quit Google for $0", not "You won't believe this").
- It must match what the clip delivers.
- No hashtags or emojis. Same language as the video.

## Titles and descriptions

- YouTube Shorts title: at most 100 characters, best under 60, the keyword in
  the first 3 words, no fake claims.
- TikTok and Instagram: 1-2 punchy sentences that tease the payoff without
  spoiling it, then 3-5 hashtags about the topic (never generic spam alone).
- Same language as the video.

## How many

- Only clips that clear the bar above. Usually 3-8 per hour of talk; a dull
  video can have 1 or none, and saying so is a valid answer.

## Report to the user

For each clip: its pieces with times, why it hooks (one line), and a score:

- 90-100: stops a cold viewer in 2 seconds AND pays off on its own
- 70-89: a strong hook OR a clear payoff, not both
- 40-69: needs context or starts slow
- 0-39: filler

Render only clips scoring 70 or more, unless the user asks otherwise.
