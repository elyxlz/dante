# Dante
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Portrait_de_Dante.jpg/640px-Portrait_de_Dante.jpg" alt="Alt Text" width="300" height="auto">
</p>

Training large autoregressive priors on speech semantic tokens rather than text for emotionally aware conversational agents.

This was my project for the Entrepreneur First AI Hackathon in London, with which I won a lunch with LinkedIn and Inflection AI co-founder Reid Hoffman.

# Process
The audio was encoded/tokenized through [Bark's](https://github.com/suno-ai/bark) quantized Hubert representations and decoded/detokenized via their hierarchical stack of coarse and fine transformers.
The semantic tokens had a resolution of 50 t/s with a vocab size of 10048.

Then a large autoregressive prior was finetuned from Llama 7B (with flash attention) with 4bit 32 rank QLora on the semantic tokens with a context length of 750 (15s).
(thanks to Katherine Crowson @alstroemeria313 for the QLora recommendation and @mahouko for the llama flash attention implementation)

The model was trained for 1 epoch on 600 hours of podcasts on 2 A100s for 6 hours (~100x real-time).

# Results
After playing with the sampling params, I achieved an output that resembled Sims language, but good enough to deceive a non-English speaker.
The model showed rudimentary prompt understanding (e.g. continuing laughs) and simple emotional intelligence (matching tone).

A simple telegram bot was made to serve the model via voice memos.

# Future
Scale up compute / data / model size until the model is able to hold a conversation.
~ 1000+ GPU hours and 50k+ hours of human interaction audio (1 gpu hour ≈ 50 audio hours)

Explore end-to-end hierarchical transformers, skipping/minimizing the AudioLM approach of splitting modeling for different data resolutions. These models should scale way more efficiently.


