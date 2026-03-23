
# engine workflow and explanation:
insert screenshot: 
++:

The mask is just an array of 1s and 0s, the same length as the logits. 
Multiply them together (via the (mask - 1) * 1e9 trick) and every 0-slot becomes -1,000,000,000 
— so cold that argmax will never pick it. 
You're not changing what the model knows, 
you're just slamming a gate shut on the tokens that would break your JSON.


The state answers the question: which gate do I apply right now? 
Without it, you'd have no way to know whether you're currently writing a function name, a parameter key, or a number value 
— and each of those requires completely different masking rules. 
The state is the engine's memory of where it is in the JSON structure.

One concrete example to make it click: 
step FUNC_NAME, you allow tokens that continue a valid function name. 
At step PARAM_VALUE for a number type, you allow only digits, ., -, and e. 
Same mask mechanism, completely different allowed set 
— the state is what switches between them.


# ressources: 
https://medium.com/@c.savur/mastering-the-logits-a-guide-to-constrained-decoding-in-hugging-face-and-vllm-357a5c1b9a28